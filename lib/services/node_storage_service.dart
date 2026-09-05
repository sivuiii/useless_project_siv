import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/stored_node_fragment.dart';

/// Node metadata storage service for Human Server memorized fragments.
/// 
/// [Architecture Requirement - Zero Persistent Plaintext]:
/// Human nodes NEVER persist fragment plaintext locally. The human brain is the
/// long-term storage mechanism. This service persists ONLY metadata (fragment ID,
/// sequence number, size in bytes, expiration, verification hash, and status)
/// to track which fragments are entrusted to and hosted by this human node.
///
/// [User Isolation Invariant]:
/// All stored metadata is strictly scoped to the active Supabase user ID.
/// User B on the same device can NEVER read, overwrite, or delete User A's
/// memorized fragment metadata.
class NodeStorageService {
  NodeStorageService._() {
    _initAuthListener();
  }
  static final NodeStorageService instance = NodeStorageService._();

  static const String _indexPrefix = 'human_server_node_index_';
  static const String _metaPrefix = 'human_server_node_metadata_';

  // Legacy un-scoped keys from earlier versions that must be purged to avoid leaking across accounts
  static const String _legacyIndexKey = 'human_server_fragment_index';
  static const String _legacyMetaPrefix = 'human_server_fragment_meta_';
  static const String _legacyPlainPrefix = 'human_server_fragment_';

  /// Strict maximum active hosted/memorized fragments allowed per human node
  static const int maxNodeCapacity = 5;
  int get maxCapacity => maxNodeCapacity;

  final ValueNotifier<List<StoredNodeFragment>> storedFragmentsNotifier =
      ValueNotifier<List<StoredNodeFragment>>([]);

  int get activeHostedCount => storedFragmentsNotifier.value.length;
  bool get isCapacityFull => activeHostedCount >= maxNodeCapacity;
  bool get isOverCapacity => activeHostedCount > maxNodeCapacity;
  bool get canReceiveFragment => activeHostedCount < maxNodeCapacity;
  int get remainingSlots => (maxNodeCapacity - activeHostedCount).clamp(0, maxNodeCapacity);

  SharedPreferences? _prefs;
  String? _overrideUserIdForTesting;
  String? _lastLoadedUserId;
  bool _authListenerInitialized = false;

  @visibleForTesting
  void setUserIdForTesting(String? userId) {
    _overrideUserIdForTesting = userId;
    _lastLoadedUserId = null;
  }

  @visibleForTesting
  void resetForTesting() {
    _prefs = null;
    _overrideUserIdForTesting = null;
    _lastLoadedUserId = null;
    storedFragmentsNotifier.value = [];
  }

  /// Dynamically resolves the currently authenticated user UUID.
  /// Never caches across auth state transitions.
  String get activeUserId {
    if (_overrideUserIdForTesting != null) {
      return _overrideUserIdForTesting!;
    }
    try {
      final user = Supabase.instance.client.auth.currentUser;
      if (user != null && user.id.isNotEmpty) {
        return user.id;
      }
    } catch (_) {}
    return 'unauthenticated';
  }

  String _indexKey(String uid) => '$_indexPrefix$uid';
  String _metadataKey(String uid, String fragmentId) => '$_metaPrefix${uid}_$fragmentId';

  void _initAuthListener() {
    if (_authListenerInitialized) return;
    try {
      Supabase.instance.client.auth.onAuthStateChange.listen((data) {
        final newUserId = data.session?.user.id;
        if (newUserId != _lastLoadedUserId) {
          if (newUserId == null) {
            onSignOut();
          } else {
            getStoredFragments(userId: newUserId);
          }
        }
      });
      _authListenerInitialized = true;
    } catch (_) {
      // Supabase may not be initialized in some test environments
    }
  }

  Future<SharedPreferences> _getPrefs() async {
    _prefs ??= await SharedPreferences.getInstance();
    return _prefs!;
  }

  /// Purges old global un-scoped keys once so that User 2 does not inherit User 1's
  /// un-scoped legacy metadata.
  Future<void> _purgeLegacyGlobalKeys(SharedPreferences prefs) async {
    if (prefs.containsKey(_legacyIndexKey)) {
      final legacyIndices = prefs.getStringList(_legacyIndexKey) ?? [];
      for (final fragId in legacyIndices) {
        await prefs.remove('$_legacyMetaPrefix$fragId');
        await prefs.remove('$_legacyPlainPrefix$fragId');
      }
      await prefs.remove(_legacyIndexKey);
    }
  }

  /// Invoked when the user logs out. Clears in-memory state so subsequent screens
  /// don't momentarily display the previous user's metadata.
  Future<void> onSignOut() async {
    _lastLoadedUserId = null;
    await Future.microtask(() {});
    storedFragmentsNotifier.value = [];
  }

  /// Loads and returns all memorized fragment metadata for the specified (or active) user.
  Future<List<StoredNodeFragment>> getStoredFragments({String? userId}) async {
    // Yield to avoid synchronous state changes during widget build phase
    await Future.microtask(() {});

    final uid = userId ?? activeUserId;
    try {
      final prefs = await _getPrefs();
      await _purgeLegacyGlobalKeys(prefs);

      final indexList = prefs.getStringList(_indexKey(uid)) ?? [];
      final List<StoredNodeFragment> fragments = [];

      for (final fragmentId in indexList) {
        final jsonStr = prefs.getString(_metadataKey(uid, fragmentId));
        if (jsonStr != null) {
          try {
            final map = jsonDecode(jsonStr) as Map<String, dynamic>;
            map.remove('plaintext'); // Enforce zero-disk-plaintext invariant
            final fragment = StoredNodeFragment.fromMap(map);
            if (!fragment.isExpired) {
              fragments.add(fragment);
            }
          } catch (e) {
            debugPrint('Error parsing stored fragment metadata $fragmentId for user $uid: $e');
          }
        }
      }

      fragments.sort((a, b) {
        final cmp = a.sequenceNumber.compareTo(b.sequenceNumber);
        if (cmp != 0) return cmp;
        return b.receivedAt.compareTo(a.receivedAt);
      });

      _lastLoadedUserId = uid;
      storedFragmentsNotifier.value = fragments;
      return fragments;
    } catch (e) {
      debugPrint('Error loading stored fragments metadata for user $uid: $e');
      return storedFragmentsNotifier.value;
    }
  }

  /// Persists metadata ONLY for a newly memorized fragment under the active user's namespace.
  Future<void> saveFragment(StoredNodeFragment fragment, {String? userId}) async {
    final uid = userId ?? activeUserId;
    final prefs = await _getPrefs();
    final jsonStr = jsonEncode(fragment.toMap());

    // Write to user-scoped metadata key
    await prefs.setString(_metadataKey(uid, fragment.fragmentId), jsonStr);

    // Update user-scoped index
    final indexList = prefs.getStringList(_indexKey(uid)) ?? [];
    if (!indexList.contains(fragment.fragmentId)) {
      indexList.add(fragment.fragmentId);
      await prefs.setStringList(_indexKey(uid), indexList);
    }

    await getStoredFragments(userId: uid);
  }

  /// Retrieves metadata for a specific fragment by its ID for the active user.
  Future<StoredNodeFragment?> getFragment(String fragmentId, {String? userId}) async {
    final uid = userId ?? activeUserId;
    final prefs = await _getPrefs();
    final jsonStr = prefs.getString(_metadataKey(uid, fragmentId));
    if (jsonStr == null) return null;

    try {
      final map = jsonDecode(jsonStr) as Map<String, dynamic>;
      map.remove('plaintext');
      return StoredNodeFragment.fromMap(map);
    } catch (e) {
      debugPrint('Error decoding fragment metadata $fragmentId for user $uid: $e');
      return null;
    }
  }

  /// Checks whether the active user currently tracks the specified fragment
  Future<bool> hasFragment(String fragmentId, {String? userId}) async {
    final uid = userId ?? activeUserId;
    final prefs = await _getPrefs();
    return prefs.containsKey(_metadataKey(uid, fragmentId));
  }

  /// Removes a fragment record from the active user's metadata tracking
  Future<void> deleteFragment(String fragmentId, {String? userId}) async {
    final uid = userId ?? activeUserId;
    final prefs = await _getPrefs();
    await prefs.remove(_metadataKey(uid, fragmentId));

    final indexList = prefs.getStringList(_indexKey(uid)) ?? [];
    if (indexList.remove(fragmentId)) {
      await prefs.setStringList(_indexKey(uid), indexList);
    }

    await getStoredFragments(userId: uid);
  }

  /// Prunes expired fragments for the active user
  Future<int> deleteExpiredFragments({String? userId}) async {
    final uid = userId ?? activeUserId;
    final prefs = await _getPrefs();
    final indexList = prefs.getStringList(_indexKey(uid)) ?? [];
    int deletedCount = 0;

    final List<String> updatedIndex = [];
    final now = DateTime.now();

    for (final fragmentId in indexList) {
      final jsonStr = prefs.getString(_metadataKey(uid, fragmentId));
      if (jsonStr != null) {
        try {
          final map = jsonDecode(jsonStr) as Map<String, dynamic>;
          final fragment = StoredNodeFragment.fromMap(map);
          if (now.isAfter(fragment.expiresAt)) {
            await prefs.remove(_metadataKey(uid, fragmentId));
            deletedCount++;
            continue;
          }
        } catch (_) {}
      }
      updatedIndex.add(fragmentId);
    }

    if (deletedCount > 0) {
      await prefs.setStringList(_indexKey(uid), updatedIndex);
      await getStoredFragments(userId: uid);
    }

    return deletedCount;
  }

  /// Clears ALL local fragment metadata records for the CURRENT user only.
  /// Does NOT touch other users' namespaces on this device.
  Future<void> clearAllMetadata({String? userId}) async {
    final uid = userId ?? activeUserId;
    final prefs = await _getPrefs();
    final indexList = prefs.getStringList(_indexKey(uid)) ?? [];

    for (final fragmentId in indexList) {
      await prefs.remove(_metadataKey(uid, fragmentId));
    }

    // Also scan all keys to ensure no orphaned fragment keys remain for THIS user
    final prefix = '$_metaPrefix${uid}_';
    final allKeys = prefs.getKeys();
    for (final key in allKeys) {
      if (key.startsWith(prefix)) {
        await prefs.remove(key);
      }
    }

    await prefs.remove(_indexKey(uid));

    if (uid == activeUserId) {
      await Future.microtask(() {});
      storedFragmentsNotifier.value = [];
    }
  }
}

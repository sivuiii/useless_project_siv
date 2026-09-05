import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/stored_node_fragment.dart';

/// Node metadata storage service for Human Server memorized fragments.
/// 
/// [Architecture Requirement - Zero Persistent Plaintext]:
/// Human nodes NEVER persist fragment plaintext locally. The human brain is the
/// long-term storage mechanism. This service persists ONLY metadata (fragment ID,
/// sequence number, size in bytes, expiration, verification hash, and status)
/// to track which fragments are entrusted to and hosted by this human node.
class NodeStorageService {
  NodeStorageService._();
  static final NodeStorageService instance = NodeStorageService._();

  static const String _storageKeyPrefix = 'human_server_fragment_meta_';
  static const String _legacyKeyPrefix = 'human_server_fragment_';
  static const String _indexKey = 'human_server_fragment_index';

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

  @visibleForTesting
  void resetForTesting() {
    _prefs = null;
    storedFragmentsNotifier.value = [];
  }

  Future<SharedPreferences> _getPrefs() async {
    _prefs ??= await SharedPreferences.getInstance();
    return _prefs!;
  }

  /// Loads and returns all memorized fragment metadata from local storage
  Future<List<StoredNodeFragment>> getStoredFragments() async {
    try {
      final prefs = await _getPrefs();
      final indexList = prefs.getStringList(_indexKey) ?? [];
      final List<StoredNodeFragment> fragments = [];

      for (final fragmentId in indexList) {
        // Try new metadata key first, then check legacy key
        var jsonStr = prefs.getString('$_storageKeyPrefix$fragmentId');
        if (jsonStr == null) {
          final legacyStr = prefs.getString('$_legacyKeyPrefix$fragmentId');
          if (legacyStr != null) {
            try {
              // Migrate legacy entry: extract metadata only and PURGE plaintext
              final map = jsonDecode(legacyStr) as Map<String, dynamic>;
              map.remove('plaintext');
              final fragment = StoredNodeFragment.fromMap(map);
              jsonStr = jsonEncode(fragment.toMap());
              // Save metadata-only under new key and remove legacy plaintext key
              await prefs.setString('$_storageKeyPrefix$fragmentId', jsonStr);
              await prefs.remove('$_legacyKeyPrefix$fragmentId');
            } catch (_) {}
          }
        }

        if (jsonStr != null) {
          try {
            final map = jsonDecode(jsonStr) as Map<String, dynamic>;
            final fragment = StoredNodeFragment.fromMap(map);
            // Ignore expired fragments
            if (!fragment.isExpired) {
              fragments.add(fragment);
            }
          } catch (e) {
            debugPrint('Error parsing stored fragment metadata $fragmentId: $e');
          }
        }
      }

      // Sort by sequence number, then received date
      fragments.sort((a, b) {
        final cmp = a.sequenceNumber.compareTo(b.sequenceNumber);
        if (cmp != 0) return cmp;
        return b.receivedAt.compareTo(a.receivedAt);
      });

      storedFragmentsNotifier.value = fragments;
      return fragments;
    } catch (e) {
      debugPrint('Error loading stored fragments metadata: $e');
      return storedFragmentsNotifier.value;
    }
  }

  /// Persists metadata ONLY for a newly memorized fragment (zero plaintext stored)
  Future<void> saveFragment(StoredNodeFragment fragment) async {
    final prefs = await _getPrefs();
    final jsonStr = jsonEncode(fragment.toMap());

    // Write to metadata key
    await prefs.setString('$_storageKeyPrefix${fragment.fragmentId}', jsonStr);
    // Explicitly ensure legacy plaintext key is deleted if it existed
    await prefs.remove('$_legacyKeyPrefix${fragment.fragmentId}');

    // Update index
    final indexList = prefs.getStringList(_indexKey) ?? [];
    if (!indexList.contains(fragment.fragmentId)) {
      indexList.add(fragment.fragmentId);
      await prefs.setStringList(_indexKey, indexList);
    }

    await getStoredFragments();
  }

  /// Retrieves metadata for a specific fragment by its ID
  Future<StoredNodeFragment?> getFragment(String fragmentId) async {
    final prefs = await _getPrefs();
    final jsonStr = prefs.getString('$_storageKeyPrefix$fragmentId');
    if (jsonStr == null) return null;

    try {
      final map = jsonDecode(jsonStr) as Map<String, dynamic>;
      return StoredNodeFragment.fromMap(map);
    } catch (e) {
      debugPrint('Error decoding fragment metadata $fragmentId: $e');
      return null;
    }
  }

  /// Checks whether this device currently tracks the specified fragment
  Future<bool> hasFragment(String fragmentId) async {
    final prefs = await _getPrefs();
    return prefs.containsKey('$_storageKeyPrefix$fragmentId') ||
        prefs.containsKey('$_legacyKeyPrefix$fragmentId');
  }

  /// Removes a fragment record from this device's metadata tracking
  Future<void> deleteFragment(String fragmentId) async {
    final prefs = await _getPrefs();
    await prefs.remove('$_storageKeyPrefix$fragmentId');
    await prefs.remove('$_legacyKeyPrefix$fragmentId');

    final indexList = prefs.getStringList(_indexKey) ?? [];
    if (indexList.remove(fragmentId)) {
      await prefs.setStringList(_indexKey, indexList);
    }

    await getStoredFragments();
  }

  /// Prunes metadata of fragments that have passed their expiration timestamp
  Future<int> deleteExpiredFragments() async {
    final prefs = await _getPrefs();
    final indexList = prefs.getStringList(_indexKey) ?? [];
    int deletedCount = 0;

    final List<String> updatedIndex = [];
    final now = DateTime.now();

    for (final fragmentId in indexList) {
      final jsonStr = prefs.getString('$_storageKeyPrefix$fragmentId');
      if (jsonStr != null) {
        try {
          final map = jsonDecode(jsonStr) as Map<String, dynamic>;
          final fragment = StoredNodeFragment.fromMap(map);
          if (now.isAfter(fragment.expiresAt)) {
            await prefs.remove('$_storageKeyPrefix$fragmentId');
            deletedCount++;
            continue;
          }
        } catch (_) {}
      }
      updatedIndex.add(fragmentId);
    }

    if (deletedCount > 0) {
      await prefs.setStringList(_indexKey, updatedIndex);
      await getStoredFragments();
    }

    return deletedCount;
  }

  /// Clears ALL local fragment metadata records maintained by this service on this device.
  /// 
  /// [Development & Testing]:
  /// Used to reset local station metadata (e.g. after a test database reset).
  /// - Clears ONLY local metadata records and keys in SharedPreferences.
  /// - Does NOT touch Supabase, user auth, remote memories, assignments, or schemas.
  /// - Does NOT store or recover any plaintext.
  /// - Immediately notifies listeners with an empty list so the UI reflects 0/5 slots used.
  Future<void> clearAllMetadata() async {
    final prefs = await _getPrefs();
    final indexList = prefs.getStringList(_indexKey) ?? [];

    for (final fragmentId in indexList) {
      await prefs.remove('$_storageKeyPrefix$fragmentId');
      await prefs.remove('$_legacyKeyPrefix$fragmentId');
    }

    // Also scan all keys to ensure no orphaned fragment keys remain
    final allKeys = prefs.getKeys();
    for (final key in allKeys) {
      if (key.startsWith(_storageKeyPrefix) || key.startsWith(_legacyKeyPrefix)) {
        await prefs.remove(key);
      }
    }

    await prefs.remove(_indexKey);
    storedFragmentsNotifier.value = [];
  }
}

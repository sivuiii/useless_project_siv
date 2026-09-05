import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/recovered_memory.dart';

/// Service managing the OWNER device's local recovery history.
/// 
/// [Architectural Distinction]:
/// - Stored node fragments: metadata only, NEVER plaintext, counts toward the 5-node-memory-slot limit.
/// - Recovered memory history: belongs to the OWNER who requested/reconstructed the memory.
///   Contains the recovered plaintext locally.
///   Does NOT count toward node storage.
///   Is NOT a StoredNodeFragment.
///   Must NEVER appear in Station / Biological Storage Vault.
///   Must NEVER be sent to other nodes automatically.
/// 
/// [Multi-User Isolation Invariant]:
/// All recovery history is strictly scoped to the authenticated Supabase user ID.
/// User B on the same device can NEVER see User A's recovered plaintext.
class RecoveryHistoryService {
  RecoveryHistoryService._() {
    _initAuthListener();
  }
  static final RecoveryHistoryService instance = RecoveryHistoryService._();

  static const String _indexPrefix = 'human_server_recovered_history_index_';
  static const String _entryPrefix = 'human_server_recovered_history_entry_';

  final ValueNotifier<List<RecoveredMemory>> historyNotifier =
      ValueNotifier<List<RecoveredMemory>>([]);

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
    historyNotifier.value = [];
  }

  /// Dynamically resolves current authenticated user ID
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
  String _entryKey(String uid, String memoryId) => '$_entryPrefix${uid}_$memoryId';

  void _initAuthListener() {
    if (_authListenerInitialized) return;
    try {
      Supabase.instance.client.auth.onAuthStateChange.listen((data) {
        final newUserId = data.session?.user.id;
        if (newUserId != _lastLoadedUserId) {
          if (newUserId == null) {
            onSignOut();
          } else {
            getHistory(userId: newUserId);
          }
        }
      });
      _authListenerInitialized = true;
    } catch (_) {
      // Supabase may not be initialized in some test environments
    }
  }

  void onSignOut() {
    _lastLoadedUserId = null;
    historyNotifier.value = [];
  }

  Future<SharedPreferences> _getPrefs() async {
    _prefs ??= await SharedPreferences.getInstance();
    return _prefs!;
  }

  /// Fetches all recovered memories in the owner's history for the active user
  Future<List<RecoveredMemory>> getHistory({String? userId}) async {
    final uid = userId ?? activeUserId;
    if (uid == 'unauthenticated') {
      historyNotifier.value = [];
      return [];
    }

    final prefs = await _getPrefs();
    final index = prefs.getStringList(_indexKey(uid)) ?? [];
    final List<RecoveredMemory> list = [];

    for (final memId in index) {
      final raw = prefs.getString(_entryKey(uid, memId));
      if (raw != null) {
        try {
          final map = jsonDecode(raw) as Map<String, dynamic>;
          list.add(RecoveredMemory.fromMap(map));
        } catch (_) {}
      }
    }

    // Sort newest first
    list.sort((a, b) => b.recoveredAt.compareTo(a.recoveredAt));
    _lastLoadedUserId = uid;
    historyNotifier.value = list;
    return list;
  }

  /// Saves a newly reconstructed memory into the owner's user-scoped history
  Future<void> saveRecoveredMemory(RecoveredMemory memory, {String? userId}) async {
    final uid = userId ?? activeUserId;
    if (uid == 'unauthenticated') return;

    final prefs = await _getPrefs();
    final idxKey = _indexKey(uid);
    final index = prefs.getStringList(idxKey) ?? [];

    if (!index.contains(memory.memoryId)) {
      index.add(memory.memoryId);
      await prefs.setStringList(idxKey, index);
    }

    await prefs.setString(
      _entryKey(uid, memory.memoryId),
      jsonEncode(memory.toMap()),
    );

    await getHistory(userId: uid);
  }

  /// Checks if a memory has already completed recovery for this user
  Future<bool> hasRecovered(String memoryId, {String? userId}) async {
    final uid = userId ?? activeUserId;
    if (uid == 'unauthenticated') return false;
    final prefs = await _getPrefs();
    final index = prefs.getStringList(_indexKey(uid)) ?? [];
    return index.contains(memoryId);
  }

  /// Fetches a specific recovered memory entry from history
  Future<RecoveredMemory?> getRecoveredMemory(String memoryId, {String? userId}) async {
    final uid = userId ?? activeUserId;
    if (uid == 'unauthenticated') return null;

    final prefs = await _getPrefs();
    final raw = prefs.getString(_entryKey(uid, memoryId));
    if (raw == null) return null;
    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return RecoveredMemory.fromMap(map);
    } catch (_) {
      return null;
    }
  }

  /// Clears only the active user's recovery history
  Future<void> clearUserHistory({String? userId}) async {
    final uid = userId ?? activeUserId;
    if (uid == 'unauthenticated') return;

    final prefs = await _getPrefs();
    final idxKey = _indexKey(uid);
    final index = prefs.getStringList(idxKey) ?? [];

    for (final memId in index) {
      await prefs.remove(_entryKey(uid, memId));
    }
    await prefs.remove(idxKey);

    if (_lastLoadedUserId == uid) {
      historyNotifier.value = [];
    }
  }
}

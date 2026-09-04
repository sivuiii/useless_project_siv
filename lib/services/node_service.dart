import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/node.dart';
import '../models/pending_delivery_fragment.dart';
import '../models/pending_recall_fragment.dart';
import 'memory_service.dart';
import 'node_storage_service.dart';

class NodeService {
  NodeService._();
  static final NodeService instance = NodeService._();

  SupabaseClient? get _supabase {
    try {
      return Supabase.instance.client;
    } catch (_) {
      return null;
    }
  }

  User? get _currentUser => _supabase?.auth.currentUser;

  /// Holds the active node record for the authenticated user and notifies listeners
  final ValueNotifier<NodeModel?> currentNodeNotifier =
      ValueNotifier<NodeModel?>(null);

  /// Ephemeral pending fragment deliveries awaiting human memorization
  final ValueNotifier<List<PendingDeliveryFragment>> pendingDeliveriesNotifier =
      ValueNotifier<List<PendingDeliveryFragment>>([]);

  /// Ephemeral pending recall requests awaiting human node retrieval response
  final ValueNotifier<List<PendingRecallFragment>> pendingRecallsNotifier =
      ValueNotifier<List<PendingRecallFragment>>([]);

  NodeModel? get currentNode => currentNodeNotifier.value;

  /// Resets the current node state (e.g. on sign out or station reset)
  void clear() {
    if (currentNodeNotifier.value != null) {
      currentNodeNotifier.value = null;
    }
    if (pendingDeliveriesNotifier.value.isNotEmpty) {
      pendingDeliveriesNotifier.value = [];
    }
    if (pendingRecallsNotifier.value.isNotEmpty) {
      pendingRecallsNotifier.value = [];
    }
    isSyncingNotifier.value = false;
  }

  /// Fetches the current user's node from Supabase, or returns the cached node.
  Future<NodeModel?> getCurrentNode({bool forceRefresh = false}) async {
    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) {
      clear();
      return null;
    }

    if (!forceRefresh && currentNode != null && currentNode!.userId == user.id) {
      return currentNode;
    }

    try {
      final data = await client
          .from('nodes')
          .select()
          .eq('user_id', user.id)
          .maybeSingle();

      if (data != null) {
        final node = NodeModel.fromMap(data);
        currentNodeNotifier.value = node;
        return node;
      }
      return null;
    } catch (e) {
      debugPrint('Error fetching node for user ${user.id}: $e');
      return currentNode;
    }
  }

  /// Registers a node for the authenticated user if one does not already exist.
  /// If it already exists, loads and returns the existing node.
  Future<NodeModel?> registerNodeIfNeeded() async {
    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) {
      clear();
      return null;
    }

    try {
      // 1. Check if node already exists for this user
      final existingData = await client
          .from('nodes')
          .select()
          .eq('user_id', user.id)
          .maybeSingle();

      if (existingData != null) {
        final node = NodeModel.fromMap(existingData);
        currentNodeNotifier.value = node;
        return node;
      }

      // 2. Insert new node record
      final now = DateTime.now().toIso8601String();
      final newNodeData = {
        'user_id': user.id,
        'status': 'online',
        'reliability': 1.0,
        'response_rate': 1.0,
        'avg_response_ms': 0,
        'last_seen': now,
      };

      final inserted = await client
          .from('nodes')
          .insert(newNodeData)
          .select()
          .single();

      final node = NodeModel.fromMap(inserted);
      currentNodeNotifier.value = node;
      return node;
    } catch (e) {
      debugPrint('Error registering/fetching node for user ${user.id}: $e');
      // If table query fails (e.g. migration pending or network issue), provide a safe in-memory fallback
      final fallbackNode = NodeModel(
        id: 'HS-TEMP-${user.id.substring(0, 8)}',
        userId: user.id,
        status: 'online',
        reliability: 1.0,
        responseRate: 1.0,
        avgResponseMs: 0,
        lastSeen: DateTime.now(),
        createdAt: DateTime.now(),
      );
      currentNodeNotifier.value = fallbackNode;
      return fallbackNode;
    }
  }

  /// Updates the node status ('online' or 'offline') in Supabase and updates last_seen
  Future<NodeModel?> updateNodeStatus(String status) async {
    final client = _supabase;
    final user = _currentUser;
    final cleanStatus = status.toLowerCase() == 'online' ? 'online' : 'offline';

    if (client == null || user == null) {
      if (currentNode != null) {
        final updated = currentNode!.copyWith(
          status: cleanStatus,
          lastSeen: DateTime.now(),
        );
        currentNodeNotifier.value = updated;
        return updated;
      }
      return null;
    }

    try {
      final now = DateTime.now().toIso8601String();
      final updatedData = await client
          .from('nodes')
          .update({
            'status': cleanStatus,
            'last_seen': now,
          })
          .eq('user_id', user.id)
          .select()
          .single();

      final node = NodeModel.fromMap(updatedData);
      currentNodeNotifier.value = node;
      return node;
    } catch (e) {
      debugPrint('Error updating node status for user ${user.id}: $e');
      // Update local state even if network call fails
      if (currentNode != null) {
        final updated = currentNode!.copyWith(
          status: cleanStatus,
          lastSeen: DateTime.now(),
        );
        currentNodeNotifier.value = updated;
        return updated;
      }
      rethrow;
    }
  }

  /// Tracks whether synchronization is currently underway
  final ValueNotifier<bool> isSyncingNotifier = ValueNotifier<bool>(false);

  /// Updates the last_seen timestamp in Supabase
  Future<void> updateLastSeen() async {
    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) return;

    try {
      final now = DateTime.now().toIso8601String();
      await client
          .from('nodes')
          .update({
            'last_seen': now,
            'status': 'online',
          })
          .eq('user_id', user.id);

      if (currentNode != null) {
        currentNodeNotifier.value = currentNode!.copyWith(
          lastSeen: DateTime.now(),
          status: 'online',
        );
      }
    } catch (e) {
      debugPrint('Error updating last_seen: $e');
    }
  }

  /// Explicit node synchronization pipeline:
  /// a. update last_seen and ensure node is online
  /// b. attempt pending fragment assignment rebalancing
  /// c. synchronize/download assigned deliveries to local storage and confirm receipts
  /// d. refresh current node telemetry state
  /// e. refresh locally stored fragments
  Future<NodeSyncResult> syncNode() async {
    // Yield to avoid synchronous state changes during widget build phase
    await Future.microtask(() {});

    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) {
      clear();
      return const NodeSyncResult(
        rebalancedCount: 0,
        fulfilledCount: 0,
        storedCount: 0,
        error: 'No authenticated user session',
      );
    }

    if (isSyncingNotifier.value) {
      final stored = await NodeStorageService.instance.getStoredFragments();
      return NodeSyncResult(
        rebalancedCount: 0,
        fulfilledCount: 0,
        pendingDeliveries: pendingDeliveriesNotifier.value,
        storedCount: stored.length,
        node: currentNode,
      );
    }

    isSyncingNotifier.value = true;
    debugPrint('NODE SYNC START');

    int rebalancedCount = 0;
    int storedCount = 0;
    List<PendingDeliveryFragment> pendingDeliveries = [];

    try {
      // Step a: update last_seen and verify status
      await updateLastSeen();

      // Step b: attempt pending fragment assignment
      debugPrint('PENDING ASSIGNMENTS CHECK');
      try {
        rebalancedCount =
            await MemoryService.instance.rebalancePendingAssignments();
        debugPrint('ASSIGNMENT RESULT: Rebalanced $rebalancedCount pending fragment assignment(s)');
      } catch (assignErr, assignStack) {
        debugPrint('PENDING ASSIGNMENT WARNING: $assignErr\n$assignStack');
      }

      // Step c: fetch pending deliveries waiting for human memorization
      pendingDeliveries =
          await MemoryService.instance.fetchPendingDeliveriesForNode();
      pendingDeliveriesNotifier.value = pendingDeliveries;

      // Step c2: fetch pending recall requests waiting for human recall
      try {
        final pendingRecalls =
            await MemoryService.instance.getPendingRecallsForNode();
        pendingRecallsNotifier.value = pendingRecalls;
      } catch (recallErr) {
        debugPrint('PENDING RECALLS WARNING: $recallErr');
      }

      // Step d: refresh current node state
      final node = await getCurrentNode(forceRefresh: true);

      // Step e: refresh locally tracked fragment metadata
      final stored = await NodeStorageService.instance.getStoredFragments();
      storedCount = stored.length;

      debugPrint('NODE SYNC COMPLETE: Memorized locally: $storedCount fragments, Pending memorization: ${pendingDeliveries.length}');

      return NodeSyncResult(
        rebalancedCount: rebalancedCount,
        fulfilledCount: 0,
        pendingDeliveries: pendingDeliveries,
        storedCount: storedCount,
        node: node,
      );
    } catch (e, stack) {
      debugPrint('NODE SYNC ERROR: $e\n$stack');
      final stored = await NodeStorageService.instance.getStoredFragments();
      return NodeSyncResult(
        rebalancedCount: rebalancedCount,
        fulfilledCount: 0,
        pendingDeliveries: pendingDeliveries,
        storedCount: stored.length,
        node: currentNode,
        error: e.toString(),
      );
    } finally {
      isSyncingNotifier.value = false;
    }
  }
}

class NodeSyncResult {
  final int rebalancedCount;
  final int fulfilledCount;
  final List<PendingDeliveryFragment> pendingDeliveries;
  final int storedCount;
  final NodeModel? node;
  final String? error;

  const NodeSyncResult({
    required this.rebalancedCount,
    required this.fulfilledCount,
    this.pendingDeliveries = const [],
    required this.storedCount,
    this.node,
    this.error,
  });

  bool get isSuccess => error == null;
}

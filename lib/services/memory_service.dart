import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/pending_delivery_fragment.dart';
import '../models/pending_recall_fragment.dart';
import '../models/retrieval_status.dart';
import '../models/stored_node_fragment.dart';
import '../models/user_stored_memory.dart';
import 'node_storage_service.dart';

class MemoryStoreResult {
  final String memoryId;
  final int fragmentCount;
  final int assignedReplicas;
  final int fulfilledReplicas;
  final int pendingReplicas;
  final List<String> assignedNodeIds;

  const MemoryStoreResult({
    required this.memoryId,
    required this.fragmentCount,
    required this.assignedReplicas,
    required this.fulfilledReplicas,
    required this.pendingReplicas,
    required this.assignedNodeIds,
  });

  bool get hasPendingReplicas => pendingReplicas > 0;
  bool get hasAssignedReplicas => assignedReplicas > 0;
  bool get hasFulfilledReplicas => fulfilledReplicas > 0;
  int get totalReplicas => assignedReplicas + fulfilledReplicas + pendingReplicas;
}

class MemoryService {
  MemoryService._();
  static final MemoryService instance = MemoryService._();

  SupabaseClient? get _supabase {
    try {
      return Supabase.instance.client;
    } catch (_) {
      return null;
    }
  }

  User? get _currentUser => _supabase?.auth.currentUser;

  /// Holds the active memories count for the authenticated user
  final ValueNotifier<int> activeMemoriesCountNotifier = ValueNotifier<int>(0);

  /// Randomized, overlapping semantic-ish token fragmentation algorithm.
  /// 
  /// [Human Server Fragmentation Model]:
  /// - Does NOT use sequential character/word slices.
  /// - Tokenizes original text into words/tokens.
  /// - Groups tokens into natural, human-memorizable semantic-ish phrases (2–3 words).
  /// - Generates overlapping multi-subset combinations (e.g. `THE MEETING + 4:30`, `4:30 + ROOM 204`).
  /// - Guarantees that for normal-length messages (>= 3 words), no single fragment contains the complete message.
  /// - Every token appears across multiple fragments to provide required redundancy for future reconstruction.
  /// - Randomizes the fragment order before transmission to ensure randomized node distribution.
  /// - Scales the number of generated fragments with text length (e.g. 4 to 40 fragments).
  static List<String> fragmentText(String text, {Random? random}) {
    final clean = text.trim().replaceAll(RegExp(r'\s+'), ' ');
    if (clean.isEmpty) return [];

    final tokens = clean.split(' ').where((t) => t.isNotEmpty).toList();
    if (tokens.isEmpty) return [];

    // Messages with 1 or 2 tokens cannot be divided into multi-subsets without single letters
    if (tokens.length <= 2) {
      return [clean];
    }

    final rng = random ?? Random();

    // 1. Partition tokens into semantic-ish phrases of 1–3 words
    final int phraseSize = tokens.length <= 4 ? 1 : 2;
    final List<String> phrases = [];
    int idx = 0;

    while (idx < tokens.length) {
      int len = phraseSize;
      // If only 1 token would remain at the end, group it with current phrase if phraseSize >= 2
      if (tokens.length - (idx + len) == 1 && phraseSize >= 2) {
        len++;
      } else if (idx + len > tokens.length) {
        len = tokens.length - idx;
      }
      phrases.add(tokens.sublist(idx, idx + len).join(' '));
      idx += len;
    }

    // Ensure we have at least 3 phrases to generate meaningful overlapping subsets
    if (phrases.length < 3 && tokens.length >= 3) {
      phrases.clear();
      for (final t in tokens) {
        phrases.add(t);
      }
    }

    final int m = phrases.length;
    final Set<String> fragmentSet = {};

    // 2. Generate overlapping multi-subset combinations
    // Determine dynamic target fragment count scaling with length
    final int targetCount = max(4, min(40, (m * 1.6).round()));

    // Linear consecutive pairs: (0, 1), (1, 2), ..., (m-2, m-1)
    // ONLY in forward direction (strictly unidirectional)
    for (int i = 0; i < m - 1; i++) {
      final p1 = phrases[i];
      final p2 = phrases[i + 1];
      fragmentSet.add('$p1 + $p2');
    }

    // Non-consecutive combinations (skip-1 and cross pairs):
    // Always added symmetrically (bidirectional) so reconstruction can unambiguously
    // distinguish true consecutive sequence transitions from subset overlap.
    if (m >= 4) {
      for (int i = 0; i < m; i++) {
        final p1 = phrases[i];
        final p2 = phrases[(i + 2) % m];
        if (p1 != p2) {
          fragmentSet.add('$p1 + $p2');
          fragmentSet.add('$p2 + $p1');
        }
      }
    } else if (m == 3) {
      fragmentSet.add('${phrases[0]} + ${phrases[2]}');
      fragmentSet.add('${phrases[2]} + ${phrases[0]}');
    }

    // If m >= 6 and more fragments needed, add 3-phrase combinations
    if (m >= 6 && fragmentSet.length < targetCount) {
      for (int i = 0; i < m && fragmentSet.length < targetCount; i++) {
        final p1 = phrases[i];
        final p2 = phrases[(i + 2) % m];
        final p3 = phrases[(i + 4) % m];
        fragmentSet.add('$p1 + $p2 + $p3');
      }
    }

    // Fill up to targetCount with randomized distinct non-adjacent phrase pairs (symmetrically)
    if (m >= 4) {
      int safety = 0;
      while (fragmentSet.length < targetCount && safety < 200) {
        safety++;
        final idx1 = rng.nextInt(m);
        var idx2 = rng.nextInt(m);
        int pickTries = 0;
        while ((idx2 == idx1 || (idx1 - idx2).abs() <= 1) && pickTries < 20) {
          idx2 = rng.nextInt(m);
          pickTries++;
        }
        if ((idx1 - idx2).abs() > 1) {
          fragmentSet.add('${phrases[idx1]} + ${phrases[idx2]}');
          fragmentSet.add('${phrases[idx2]} + ${phrases[idx1]}');
        }
      }
    }

    final List<String> result = fragmentSet.toList();

    // 3. Strict Non-Completeness Guarantee:
    // Ensure no fragment ever contains the entire original message for tokens.length >= 3
    final originalTokenSet = tokens.map((t) => t.toUpperCase()).toSet();
    result.removeWhere((frag) {
      final fragTokens = frag
          .replaceAll('+', ' ')
          .split(RegExp(r'\s+'))
          .where((t) => t.isNotEmpty)
          .map((t) => t.toUpperCase())
          .toSet();
      return fragTokens.containsAll(originalTokenSet);
    });

    // 4. Randomize fragment order
    result.shuffle(rng);

    return result;
  }

  /// Calculates sha256 checksum for a text fragment
  static String calculateHash(String text) {
    final bytes = utf8.encode(text);
    return sha256.convert(bytes).toString();
  }

  /// Reconstructs the original message from recalled fragments using the overlapping phrase model.
  static String reconstructFromFragments(List<String> fragments) {
    final cleanFragments = fragments
        .map((f) => f.trim())
        .where((f) => f.isNotEmpty)
        .toList();

    if (cleanFragments.isEmpty) return '';

    // If single fragment without '+', return it directly
    if (cleanFragments.length == 1 && !cleanFragments.first.contains('+')) {
      return cleanFragments.first;
    }

    // Check if all fragments have no '+' (e.g. short messages)
    if (cleanFragments.every((f) => !f.contains('+'))) {
      return cleanFragments.first;
    }

    // Parse fragments into ordered phrase tuples
    final Set<String> allPhrases = {};
    final Set<String> directPairs = {};

    for (final frag in cleanFragments) {
      if (frag.contains('+')) {
        final parts = frag
            .split('+')
            .map((p) => p.trim())
            .where((p) => p.isNotEmpty)
            .toList();
        for (int i = 0; i < parts.length; i++) {
          allPhrases.add(parts[i]);
          if (i + 1 < parts.length) {
            directPairs.add('${parts[i]}|||${parts[i + 1]}');
          }
        }
      } else {
        allPhrases.add(frag);
      }
    }

    if (allPhrases.isEmpty) return '';
    if (allPhrases.length == 1) return allPhrases.first;

    // Filter to ONLY unidirectional transitions:
    // Consecutive edges A -> B are unidirectional (B -> A never exists).
    // Skip/cross edges A -> C are bidirectional (C -> A also exists).
    final Map<String, String> forward = {};
    final Map<String, String> backward = {};

    for (final pair in directPairs) {
      final split = pair.split('|||');
      final from = split[0];
      final to = split[1];
      final reverse = '$to|||$from';

      if (!directPairs.contains(reverse)) {
        forward[from] = to;
        backward[to] = from;
      }
    }

    // Find the head of the chain: node with no incoming unidirectional edge
    String? startNode;
    for (final phrase in allPhrases) {
      if (!backward.containsKey(phrase) && forward.containsKey(phrase)) {
        startNode = phrase;
        break;
      }
    }

    // If all unidirectional edges form a closed cycle (e.g. legacy fragments with wrap),
    // pick the node starting with capital/sentence start or first phrase
    startNode ??= allPhrases.firstWhere(
      (p) => forward.containsKey(p),
      orElse: () => allPhrases.first,
    );

    // Follow forward chain
    final List<String> chain = [startNode];
    final Set<String> visited = {startNode};

    while (forward.containsKey(chain.last)) {
      final next = forward[chain.last]!;
      if (visited.contains(next)) break;
      chain.add(next);
      visited.add(next);
    }

    // Add any remaining phrases that were not connected
    for (final phrase in allPhrases) {
      if (!visited.contains(phrase)) {
        chain.add(phrase);
      }
    }

    return chain.join(' ').replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  /// Creates a memory record:
  /// 1. Client-side fragmenting of plaintext.
  /// 2. Calculates fragment checksum hashes.
  /// 3. Server enforces 6-month expiry and creates metadata records.
  /// 4. Plaintext is only routed to temporary delivery inboxes for assigned peer nodes.
  Future<MemoryStoreResult> createMemory(String text) async {
    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) {
      throw const AuthException('You must be signed in to store memories.');
    }

    final trimmed = text.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError('Memory text cannot be empty.');
    }

    // 1. Enforce active memories limit (max 5)
    final activeCount = await getActiveMemoriesCount(forceRefresh: true);
    if (activeCount >= 5) {
      throw StateError(
        'Active storage limit reached: each user can have up to 5 active memories.',
      );
    }

    // 2. Fragment text client-side
    final fragments = fragmentText(trimmed);
    if (fragments.isEmpty) {
      throw ArgumentError('Unable to generate fragments from provided text.');
    }

    try {
      // 4. Call the secure RPC
      // Canonical SHA-256 hash is computed server-side with pgcrypto.
      // Server enforces 6-month expiry & metadata-only storage.
      final response = await client.rpc(
        'store_memory_with_fragments',
        params: {
          'p_fragments': fragments,
        },
      );

      final data = response is Map<String, dynamic>
          ? response
          : Map<String, dynamic>.from(response as Map);

      final result = MemoryStoreResult(
        memoryId: data['memory_id'] as String? ?? '',
        fragmentCount:
            (data['fragment_count'] as num?)?.toInt() ?? fragments.length,
        assignedReplicas:
            (data['assigned_replicas'] as num?)?.toInt() ?? 0,
        fulfilledReplicas:
            (data['fulfilled_replicas'] as num?)?.toInt() ?? 0,
        pendingReplicas: (data['pending_replicas'] as num?)?.toInt() ??
            (fragments.length * 3),
        assignedNodeIds: (data['assigned_nodes'] as List<dynamic>?)
                ?.map((e) => e.toString())
                .toList() ??
            [],
      );

      // Refresh memory count
      await getActiveMemoriesCount(forceRefresh: true);

      return result;
    } catch (e) {
      debugPrint('Error in store_memory_with_fragments: $e');
      rethrow;
    }
  }

  /// Rebalances pending fragment assignments to available online peer nodes
  Future<int> rebalancePendingAssignments() async {
    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) {
      debugPrint('[PENDING ASSIGNMENTS] No client or authenticated user session');
      return 0;
    }

    try {
      debugPrint('PENDING ASSIGNMENTS CHECK: Calling assign_pending_fragments_for_available_nodes RPC');
      final response =
          await client.rpc('assign_pending_fragments_for_available_nodes');
      final data = response is Map<String, dynamic>
          ? response
          : Map<String, dynamic>.from(response as Map);
      final count = (data['rebalanced_count'] as num?)?.toInt() ?? 0;
      debugPrint('ASSIGNMENT RESULT: Rebalanced $count pending assignment(s) to peer nodes');
      return count;
    } catch (e, stack) {
      debugPrint('ASSIGNMENT RESULT ERROR: Failed to assign pending fragments: $e\n$stack');
      rethrow;
    }
  }

  /// Fetches pending deliveries from the temporary transport inbox without saving them to disk.
  /// Plaintext exists strictly in transient memory in the returned PendingDeliveryFragment objects.
  Future<List<PendingDeliveryFragment>> fetchPendingDeliveriesForNode() async {
    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) return [];

    try {
      debugPrint('DELIVERY CHECK: Fetching pending deliveries from transport inbox for node...');
      final response = await client.rpc('get_pending_deliveries_for_node');
      final list = (response as List<dynamic>?) ?? [];
      debugPrint('DELIVERY CHECK: Found ${list.length} pending delivery item(s)');

      final List<PendingDeliveryFragment> pendingList = [];

      for (final item in list) {
        final map = Map<String, dynamic>.from(item as Map);
        final delivery = PendingDeliveryFragment.fromMap(map);

        // Verify fragment integrity against canonical SHA-256 hash
        final actualHash = calculateHash(delivery.payloadText);
        if (delivery.expectedHash.isNotEmpty &&
            actualHash.trim().toLowerCase() !=
                delivery.expectedHash.trim().toLowerCase()) {
          debugPrint(
              'HASH REJECTED: Hash mismatch for fragment ${delivery.fragmentId}! Expected: ${delivery.expectedHash}, Actual: $actualHash');
          continue;
        }

        // Filter out if this fragment is already memorized in local storage
        if (await NodeStorageService.instance.hasFragment(delivery.fragmentId)) {
          debugPrint(
              'DELIVERY FILTERED: Fragment ${delivery.fragmentId} is already memorized locally.');
          continue;
        }

        // Filter out if node is already at full capacity
        if (NodeStorageService.instance.isCapacityFull) {
          debugPrint(
              'DELIVERY FILTERED: Node is at maximum capacity (${NodeStorageService.instance.activeHostedCount}/5).');
          continue;
        }

        debugPrint('HASH VERIFIED: Canonical SHA-256 matches payload for fragment ${delivery.fragmentId}');
        pendingList.add(delivery);
      }

      return pendingList;
    } catch (e, stack) {
      debugPrint('DELIVERY ERROR: Error fetching pending deliveries for node: $e\n$stack');
      rethrow;
    }
  }

  /// Confirms that the human user has read and memorized the fragment.
  /// 
  /// [Crucial Invariant - Order of Operations]:
  /// 1. Calls Supabase confirm_fragment_receipt to mark memorized/fulfilled and delete transport inbox copy.
  /// 2. Only upon successful server confirmation, persists fragment METADATA ONLY to local storage (zero plaintext persisted).
  /// 3. Only upon successful server confirmation, wipes the ephemeral plaintext from client memory.
  /// Plaintext is NEVER cleared and local metadata is NEVER stored before successful server receipt confirmation.
  Future<void> confirmMemorizedFragment(PendingDeliveryFragment delivery) async {
    final client = _supabase;
    if (client == null) {
      throw const AuthException('Supabase client not initialized or no network connection.');
    }

    if (delivery.assignmentId.trim().isEmpty) {
      throw ArgumentError('Invalid delivery: assignmentId cannot be empty.');
    }

    // Step 1: Confirm receipt on Supabase first (marks memorized/fulfilled and purges server transport plaintext).
    // Uses a 15-second timeout to prevent the UI from hanging indefinitely in a loading state.
    debugPrint('RECEIPT CONFIRMATION: Calling confirm_fragment_receipt for assignment ${delivery.assignmentId}...');
    final response = await client.rpc(
      'confirm_fragment_receipt',
      params: {'p_assignment_id': delivery.assignmentId},
    ).timeout(
      const Duration(seconds: 15),
      onTimeout: () {
        throw TimeoutException(
          'Confirmation request timed out after 15 seconds. The server did not respond.',
        );
      },
    );

    if (response != true) {
      throw StateError(
        'Server failed to confirm fragment receipt (server response: $response).',
      );
    }
    debugPrint('RECEIPT CONFIRMED: Assignment ${delivery.assignmentId} confirmed, server transport inbox row deleted');

    // Step 2: ONLY upon successful server confirmation, persist metadata ONLY (no plaintext)
    final actualHash = calculateHash(delivery.payloadText);
    final storedMeta = StoredNodeFragment(
      fragmentId: delivery.fragmentId,
      memoryId: delivery.memoryId,
      sequenceNumber: delivery.sequenceNumber,
      sizeBytes: delivery.sizeBytes > 0
          ? delivery.sizeBytes
          : delivery.payloadText.codeUnits.length,
      receivedAt: DateTime.now(),
      expiresAt: delivery.expiresAt,
      assignmentId: delivery.assignmentId,
      verificationHash: actualHash,
      status: 'MEMORIZED',
    );

    await NodeStorageService.instance.saveFragment(storedMeta);
    debugPrint('METADATA SAVED: Fragment ${delivery.fragmentId} metadata stored locally (no plaintext)');

    // Step 3: ONLY upon successful server confirmation, wipe ephemeral plaintext from RAM
    delivery.clear();
    debugPrint('MEMORY WIPED: Ephemeral plaintext cleared from device memory for fragment ${delivery.fragmentId}');
  }

  /// Synchronizes incoming fragment deliveries for this node from the temporary transport inbox.
  /// Returns the list of pending deliveries waiting for human memorization.
  Future<int> syncDeliveriesForNode() async {
    final pending = await fetchPendingDeliveriesForNode();
    return pending.length;
  }

  /// Evaluates whether a human node can accept an additional fragment assignment
  /// based on the strict 5-fragment capacity rule.
  static bool canNodeAcceptFragment(int currentHostedCount) {
    return currentHostedCount < NodeStorageService.maxNodeCapacity;
  }

  /// Forgets a successfully recalled fragment:
  /// 1. Calls Supabase RPC 'retire_recalled_fragment_assignment' to transition
  ///    the server assignment status to 'recalled', freeing the node's hosting slot.
  /// 2. Deletes the node's local metadata-only record from NodeStorageService.
  /// 
  /// [Crucial Invariant]:
  /// If the server retrieval/confirmation fails, the local metadata is NOT deleted,
  /// preserving custody so the fragment is not lost.
  Future<void> forgetRecalledFragment({
    required String assignmentId,
    required String fragmentId,
  }) async {
    final client = _supabase;
    if (client == null) {
      throw const AuthException('Supabase client not initialized or no connection.');
    }

    if (assignmentId.trim().isEmpty) {
      throw ArgumentError('assignmentId cannot be empty.');
    }

    // Step 1: Server confirmation first
    debugPrint('RECALL RETIREMENT: Calling retire_recalled_fragment_assignment for $assignmentId...');
    final response = await client.rpc(
      'retire_recalled_fragment_assignment',
      params: {'p_assignment_id': assignmentId},
    ).timeout(
      const Duration(seconds: 15),
      onTimeout: () {
        throw TimeoutException('Server timed out while confirming fragment recall retirement.');
      },
    );

    if (response != true) {
      throw StateError('Server failed to retire fragment assignment (response: $response)');
    }
    debugPrint('RECALL CONFIRMED: Server retired assignment $assignmentId (capacity slot freed)');

    // Step 2: ONLY upon verified server confirmation, forget local metadata
    await NodeStorageService.instance.deleteFragment(fragmentId);
    debugPrint('FRAGMENT FORGOTTEN: Recalled fragment $fragmentId purged from local biological vault.');
  }

  /// Fetches count of active memories for current user
  Future<int> getActiveMemoriesCount({bool forceRefresh = false}) async {
    final client = _supabase;
    final user = _currentUser;
    if (client == null || user == null) {
      activeMemoriesCountNotifier.value = 0;
      return 0;
    }

    try {
      final response = await client
          .from('memories')
          .select()
          .eq('owner_user_id', user.id)
          .eq('status', 'active');

      final count = (response as List).length;
      activeMemoriesCountNotifier.value = count;
      return count;
    } catch (e) {
      debugPrint('Error getting active memories count: $e');
      return activeMemoriesCountNotifier.value;
    }
  }

  /// Lists active memories stored by the current user for retrieval selection
  Future<List<UserStoredMemory>> getUserStoredMemories() async {
    final client = _supabase;
    if (client == null || _currentUser == null) return [];

    try {
      final response = await client.rpc('get_user_stored_memories');
      final list = (response as List<dynamic>?) ?? [];
      return list
          .map((item) => UserStoredMemory.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList();
    } catch (e) {
      debugPrint('Error fetching user stored memories: $e');
      return [];
    }
  }

  /// Creates or gets an open retrieval request for a stored memory
  Future<String> initiateMemoryRetrieval(String memoryId) async {
    final client = _supabase;
    if (client == null || _currentUser == null) {
      throw const AuthException('You must be signed in to initiate retrieval.');
    }

    final trimmed = memoryId.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError('Memory ID cannot be empty.');
    }

    final response = await client.rpc(
      'initiate_memory_retrieval',
      params: {'p_memory_id': trimmed},
    );

    final map = Map<String, dynamic>.from(response as Map);
    return map['retrieval_id'] as String;
  }

  /// Fetches the status of a retrieval request and all collected recall responses so far
  Future<RetrievalStatusResult> getRetrievalStatusAndFragments(String retrievalId) async {
    final client = _supabase;
    if (client == null || _currentUser == null) {
      throw const AuthException('You must be signed in to check retrieval status.');
    }

    final response = await client.rpc(
      'get_retrieval_status_and_fragments',
      params: {'p_retrieval_id': retrievalId},
    );

    final map = Map<String, dynamic>.from(response as Map);
    return RetrievalStatusResult.fromMap(map);
  }

  /// Completes a retrieval session and immediately PURGES all temporary recall plaintext rows from the database
  Future<void> completeMemoryRetrieval(String retrievalId) async {
    final client = _supabase;
    if (client == null || _currentUser == null) return;

    try {
      await client.rpc(
        'complete_memory_retrieval',
        params: {'p_retrieval_id': retrievalId},
      );
      debugPrint('RETRIEVAL COMPLETED: Server temporary recall plaintext buffer purged for $retrievalId');
    } catch (e) {
      debugPrint('Error completing memory retrieval: $e');
    }
  }

  /// Fetches pending recall requests directed to this human node
  Future<List<PendingRecallFragment>> getPendingRecallsForNode() async {
    final client = _supabase;
    if (client == null || _currentUser == null) return [];

    try {
      final response = await client.rpc('get_pending_recalls_for_node');
      final list = (response as List<dynamic>?) ?? [];
      return list
          .map((item) => PendingRecallFragment.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList();
    } catch (e) {
      debugPrint('Error fetching pending recalls for node: $e');
      return [];
    }
  }

  /// Submits the human node's recalled plaintext:
  /// 1. Verifies SHA-256 hash server-side.
  /// 2. Stores temporary transport response row for the owner.
  /// 3. Retires the assignment on the server to 'recalled' (frees memory slot).
  /// 4. ONLY upon server acceptance, deletes local metadata record from this node.
  Future<bool> submitFragmentRecall({
    required String retrievalId,
    required String assignmentId,
    required String fragmentId,
    required String recalledText,
  }) async {
    final client = _supabase;
    if (client == null || _currentUser == null) {
      throw const AuthException('Supabase client not initialized or no network connection.');
    }

    final trimmed = recalledText.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError('Recalled plaintext cannot be empty.');
    }

    debugPrint('SUBMITTING RECALL: Submitting recall for assignment $assignmentId...');
    final response = await client.rpc(
      'submit_fragment_recall',
      params: {
        'p_retrieval_id': retrievalId,
        'p_assignment_id': assignmentId,
        'p_recalled_text': trimmed,
      },
    ).timeout(
      const Duration(seconds: 15),
      onTimeout: () {
        throw TimeoutException('Server timed out while submitting fragment recall.');
      },
    );

    final map = Map<String, dynamic>.from(response as Map);
    if (map['success'] != true) {
      throw StateError('Server rejected recall submission: ${map['message']}');
    }

    // Server accepted recall and marked assignment 'recalled'
    // Step 2: Purge local metadata from this station (frees slot in UI)
    await NodeStorageService.instance.deleteFragment(fragmentId);
    debugPrint('RECALL COMPLETE: Fragment $fragmentId forgotten locally (slot freed).');

    return true;
  }
}

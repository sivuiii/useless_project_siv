import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/pending_delivery_fragment.dart';
import '../models/pending_recall_fragment.dart';
import '../models/recovered_memory.dart';
import '../models/retrieval_status.dart';
import '../models/stored_node_fragment.dart';
import '../models/user_stored_memory.dart';
import 'node_storage_service.dart';
import 'recovery_history_service.dart';

class FragmentPayload {
  final String text;
  final List<int> canonicalIndices;
  final int sequenceNumber;

  const FragmentPayload({
    required this.text,
    required this.canonicalIndices,
    required this.sequenceNumber,
  });

  Map<String, dynamic> toJson() => {
    'text': text,
    'indices': canonicalIndices,
    'sequence_number': sequenceNumber,
  };
}

class MemoryStoreResult {
  final String memoryId;
  final int packetCount;
  final int packetsAssigned;
  final int packetsPending;
  final int packetsMemorized;
  final String status;

  const MemoryStoreResult({
    required this.memoryId,
    required this.packetCount,
    required this.packetsAssigned,
    required this.packetsPending,
    required this.packetsMemorized,
    required this.status,
  });

  int get fragmentCount => packetCount;
  int get assignedReplicas => packetsAssigned;
  int get fulfilledReplicas => packetsMemorized;
  int get pendingReplicas => packetsPending;
  List<String> get assignedNodeIds => const [];
  bool get hasPendingReplicas => packetsPending > 0;
  bool get hasAssignedReplicas => packetsAssigned > 0;
  bool get hasFulfilledReplicas => packetsMemorized > 0;
  int get totalReplicas => packetCount;
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

  /// Generates randomized overlapping fragments with canonical phrase indices metadata.
  /// Canonical indices preserve exact position/overlap identity without storing plaintext.
  static List<FragmentPayload> generateFragmentPayloads(String text, {Random? random}) {
    final clean = text.trim().replaceAll(RegExp(r'\s+'), ' ');
    if (clean.isEmpty) return [];

    final tokens = clean.split(' ').where((t) => t.isNotEmpty).toList();
    if (tokens.isEmpty) return [];

    if (tokens.length <= 2) {
      return [
        FragmentPayload(
          text: clean,
          canonicalIndices: const [0],
          sequenceNumber: 1,
        )
      ];
    }

    final rng = random ?? Random();

    // 1. Partition tokens into semantic-ish phrases of 1–3 words
    final int phraseSize = tokens.length <= 4 ? 1 : 2;
    final List<String> phrases = [];
    int idx = 0;

    while (idx < tokens.length) {
      int len = phraseSize;
      if (tokens.length - (idx + len) == 1 && phraseSize >= 2) {
        len++;
      } else if (idx + len > tokens.length) {
        len = tokens.length - idx;
      }
      phrases.add(tokens.sublist(idx, idx + len).join(' '));
      idx += len;
    }

    if (phrases.length < 3 && tokens.length >= 3) {
      phrases.clear();
      for (final t in tokens) {
        phrases.add(t);
      }
    }

    final int m = phrases.length;
    final Map<String, List<int>> fragmentMap = {};

    void addFragment(List<int> phraseIndices) {
      final fragText = phraseIndices.map((i) => phrases[i]).join(' + ');
      fragmentMap[fragText] = phraseIndices;
    }

    final int targetCount = max(4, min(40, (m * 1.6).round()));

    // Linear consecutive pairs: (0, 1), (1, 2), ..., (m-2, m-1)
    for (int i = 0; i < m - 1; i++) {
      addFragment([i, i + 1]);
    }

    // Non-consecutive combinations (skip-1 and cross pairs)
    if (m >= 4) {
      for (int i = 0; i < m; i++) {
        final j = (i + 2) % m;
        if (phrases[i] != phrases[j]) {
          addFragment([i, j]);
          addFragment([j, i]);
        }
      }
    } else if (m == 3) {
      addFragment([0, 2]);
      addFragment([2, 0]);
    }

    // 3-phrase combinations if m >= 6
    if (m >= 6 && fragmentMap.length < targetCount) {
      for (int i = 0; i < m && fragmentMap.length < targetCount; i++) {
        addFragment([i, (i + 2) % m, (i + 4) % m]);
      }
    }

    // Randomized distinct pairs
    if (m >= 4) {
      int safety = 0;
      while (fragmentMap.length < targetCount && safety < 200) {
        safety++;
        final idx1 = rng.nextInt(m);
        var idx2 = rng.nextInt(m);
        int pickTries = 0;
        while ((idx2 == idx1 || (idx1 - idx2).abs() <= 1) && pickTries < 20) {
          idx2 = rng.nextInt(m);
          pickTries++;
        }
        if ((idx1 - idx2).abs() > 1) {
          addFragment([idx1, idx2]);
          addFragment([idx2, idx1]);
        }
      }
    }

    // Strict Non-Completeness Guarantee:
    // No single fragment may contain all original tokens
    final originalTokenSet = tokens.map((t) => t.toUpperCase()).toSet();
    final keys = fragmentMap.keys.toList();
    for (final frag in keys) {
      final fragTokens = frag
          .replaceAll('+', ' ')
          .split(RegExp(r'\s+'))
          .where((t) => t.isNotEmpty)
          .map((t) => t.toUpperCase())
          .toSet();
      if (fragTokens.containsAll(originalTokenSet)) {
        fragmentMap.remove(frag);
      }
    }

    final shuffledKeys = fragmentMap.keys.toList()..shuffle(rng);
    final List<FragmentPayload> payloads = [];
    for (int i = 0; i < shuffledKeys.length; i++) {
      final k = shuffledKeys[i];
      payloads.add(
        FragmentPayload(
          text: k,
          canonicalIndices: fragmentMap[k]!,
          sequenceNumber: i + 1,
        ),
      );
    }

    return payloads;
  }

  /// Returns randomized overlapping fragments as strings
  static List<String> fragmentText(String text, {Random? random}) {
    return generateFragmentPayloads(text, random: random).map((p) => p.text).toList();
  }

  /// Calculates sha256 checksum for a text fragment
  static String calculateHash(String text) {
    final bytes = utf8.encode(text);
    return sha256.convert(bytes).toString();
  }

  /// Reconstructs the original message from recalled responses using overlapping observations
  /// and canonical slot consensus.
  /// 
  /// In the Human Server model:
  /// - Humans are the storage (no server-side canonical plaintext verification).
  /// - Responses can arrive in any order, with duplicate submissions, redundant overlaps,
  ///   or missing fragments.
  /// - Consensus voting across overlapping observations detects and filters out isolated
  ///   human typos or inconsistent outlier inputs.
  /// - Deterministically produces the exact ordered message without duplicate words.
  static String reconstructFromResponses(List<RecalledFragmentItem> items) {
    if (items.isEmpty) return '';

    final Map<int, List<String>> slotObservations = {};

    for (final item in items) {
      final text = item.recalledText.trim();
      if (text.isEmpty) continue;

      final parts = _parseRecalledParts(text);
      final indices = item.canonicalIndices;

      if (indices.isNotEmpty && parts.isNotEmpty) {
        for (int i = 0; i < parts.length && i < indices.length; i++) {
          final slot = indices[i];
          slotObservations.putIfAbsent(slot, () => []).add(parts[i]);
        }
      } else if (indices.isNotEmpty) {
        slotObservations.putIfAbsent(indices.first, () => []).add(text);
      }
    }

    if (slotObservations.isNotEmpty) {
      final sortedSlots = slotObservations.keys.toList()..sort();
      final List<String> orderedPhrases = [];

      for (final slot in sortedSlots) {
        final observations = slotObservations[slot]!;
        final consensusPhrase = _selectConsensusPhrase(observations);
        if (consensusPhrase.isNotEmpty) {
          orderedPhrases.add(consensusPhrase);
        }
      }

      return orderedPhrases.join(' ').trim();
    }

    // Fallback if no canonical indices were present
    return reconstructFromFragments(items.map((i) => i.recalledText).toList());
  }

  /// Parses a recalled fragment string into its constituent phrase parts.
  static List<String> _parseRecalledParts(String text) {
    return text
        .split(RegExp(r'\s*[\+;,]\s*'))
        .map((p) => p.trim())
        .where((p) => p.isNotEmpty)
        .toList();
  }

  /// Resolves overlapping human observations for a single canonical slot.
  /// Detects and eliminates outliers, typos, and partial phrases via consensus voting.
  static String _selectConsensusPhrase(List<String> observations) {
    if (observations.isEmpty) return '';
    if (observations.length == 1) return observations.first;

    // 1. Group observations by uppercase normalized string
    final Map<String, int> counts = {};
    final Map<String, String> originalVariants = {};

    for (final raw in observations) {
      final norm = raw.trim().replaceAll(RegExp(r'\s+'), ' ').toUpperCase();
      counts[norm] = (counts[norm] ?? 0) + 1;
      originalVariants.putIfAbsent(norm, () => raw.trim());
    }

    if (counts.length == 1) {
      return originalVariants[counts.keys.first]!;
    }

    // 2. Score candidates using vote count + cross-support (substring & near-typo)
    String bestCandidate = counts.keys.first;
    double highestScore = -1.0;

    for (final candidate in counts.keys) {
      final voteCount = counts[candidate]!;
      double score = voteCount.toDouble();

      for (final other in counts.keys) {
        if (candidate == other) continue;
        final otherCount = counts[other]!;

        // Substring support: e.g. "MEETING" partially supports "THE MEETING"
        if (candidate.contains(other)) {
          score += otherCount * 0.7;
        } else if (_isNearTypo(candidate, other)) {
          // If other is a typo (e.g. "THE METING" vs "THE MEETING"),
          // the more frequent candidate absorbs support
          if (voteCount > otherCount) {
            score += otherCount * 0.5;
          }
        }
      }

      if (score > highestScore) {
        highestScore = score;
        bestCandidate = candidate;
      }
    }

    return originalVariants[bestCandidate] ?? bestCandidate;
  }

  /// Checks if two strings are within a small edit distance (likely typo)
  static bool _isNearTypo(String s1, String s2) {
    if ((s1.length - s2.length).abs() > 2) return false;
    int diff = 0;
    int i = 0, j = 0;
    while (i < s1.length && j < s2.length) {
      if (s1[i] != s2[j]) {
        diff++;
        if (diff > 2) return false;
        if (s1.length > s2.length) {
          i++;
        } else if (s2.length > s1.length) {
          j++;
        } else {
          i++;
          j++;
        }
      } else {
        i++;
        j++;
      }
    }
    diff += (s1.length - i) + (s2.length - j);
    return diff <= 2;
  }


  /// Reconstructs the original message from recalled fragments using the overlapping phrase model.
  static String reconstructFromFragments(
    List<String> fragments, {
    List<List<int>>? fragmentIndices,
  }) {
    final cleanFragments = fragments
        .map((f) => f.trim())
        .where((f) => f.isNotEmpty)
        .toList();

    if (cleanFragments.isEmpty) return '';

    if (fragmentIndices != null && fragmentIndices.isNotEmpty) {
      final Map<int, String> phraseMap = {};
      for (int i = 0; i < cleanFragments.length && i < fragmentIndices.length; i++) {
        final frag = cleanFragments[i];
        final indices = fragmentIndices[i];
        if (indices.isNotEmpty && frag.contains('+')) {
          final parts = frag
              .split('+')
              .map((p) => p.trim())
              .where((p) => p.isNotEmpty)
              .toList();
          for (int j = 0; j < parts.length && j < indices.length; j++) {
            phraseMap[indices[j]] = parts[j];
          }
        } else if (indices.isNotEmpty) {
          phraseMap[indices.first] = frag;
        }
      }
      if (phraseMap.isNotEmpty) {
        final sortedIndices = phraseMap.keys.toList()..sort();
        return sortedIndices.map((idx) => phraseMap[idx]!).join(' ').trim();
      }
    }

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

    // Filter to ONLY unidirectional transitions
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

    String? startNode;
    for (final phrase in allPhrases) {
      if (!backward.containsKey(phrase) && forward.containsKey(phrase)) {
        startNode = phrase;
        break;
      }
    }

    startNode ??= allPhrases.firstWhere(
      (p) => forward.containsKey(p),
      orElse: () => allPhrases.first,
    );

    final List<String> chain = [startNode];
    final Set<String> visited = {startNode};

    while (forward.containsKey(chain.last)) {
      final next = forward[chain.last]!;
      if (visited.contains(next)) break;
      chain.add(next);
      visited.add(next);
    }

    for (final phrase in allPhrases) {
      if (!visited.contains(phrase)) {
        chain.add(phrase);
      }
    }

    return chain.join(' ').replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  /// Creates a memory record:
  /// 1. Client-side fragmenting into FragmentPayloads.
  /// 2. Clear transient plaintext references immediately.
  /// 3. Server enforces 1 fragment per distinct human node and 5 active capacity cap.
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

    // 2. Fragment text client-side into FragmentPayloads
    final payloads = generateFragmentPayloads(trimmed);
    if (payloads.isEmpty) {
      throw ArgumentError('Unable to generate fragments from provided text.');
    }

    try {
      // 3. Call the secure RPC
      final response = await client.rpc(
        'store_memory_with_fragments',
        params: {
          'p_fragments': payloads.map((p) => p.toJson()).toList(),
        },
      );

      final data = response is Map<String, dynamic>
          ? response
          : Map<String, dynamic>.from(response as Map);

      final result = MemoryStoreResult(
        memoryId: data['memory_id'] as String? ?? '',
        packetCount: (data['packet_count'] as num?)?.toInt() ?? payloads.length,
        packetsAssigned: (data['packets_assigned'] as num?)?.toInt() ?? 0,
        packetsPending: (data['packets_pending'] as num?)?.toInt() ?? 0,
        packetsMemorized: (data['packets_memorized'] as num?)?.toInt() ?? 0,
        status: data['status'] as String? ?? 'active',
      );

      // Refresh memory count
      await getActiveMemoriesCount(forceRefresh: true);

      return result;
    } catch (e) {
      debugPrint('Error in store_memory_with_fragments: $e');
      rethrow;
    }
  }

  /// Fetches live memorized packet status for sender without exposing recipients or plaintext
  Future<Map<String, dynamic>> getMemoryPacketStatus(String memoryId) async {
    final client = _supabase;
    if (client == null || _currentUser == null) return {};

    try {
      final response = await client.rpc(
        'get_memory_packet_status',
        params: {'p_memory_id': memoryId},
      );
      return response is Map<String, dynamic>
          ? response
          : Map<String, dynamic>.from(response as Map);
    } catch (e) {
      debugPrint('Error fetching memory packet status: $e');
      return {};
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

  /// Lists active, unrecovered memories stored by the current user for retrieval selection
  Future<List<UserStoredMemory>> getUserStoredMemories() async {
    final client = _supabase;
    if (client == null || _currentUser == null) return [];

    try {
      final response = await client.rpc('get_user_stored_memories');
      final list = (response as List<dynamic>?) ?? [];
      final memories = list
          .map((item) => UserStoredMemory.fromMap(Map<String, dynamic>.from(item as Map)))
          .where((m) => !m.isCompleted)
          .toList();

      // Filter out any memories already in the local recovery history
      final List<UserStoredMemory> filtered = [];
      for (final mem in memories) {
        final alreadyRecovered =
            await RecoveryHistoryService.instance.hasRecovered(mem.id);
        if (!alreadyRecovered) {
          filtered.add(mem);
        }
      }
      return filtered;
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

    // Client-side guard: verify memory has not already completed recovery
    if (await RecoveryHistoryService.instance.hasRecovered(trimmed)) {
      throw StateError(
        'This memory has already completed recovery and cannot be recovered again.',
      );
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

  /// Completes a retrieval session:
  /// 1. Optionally saves the reconstructed plaintext into the owner device's local recovery history.
  /// 2. Calls Supabase RPC to transition memory to 'recovered' and immediately PURGES all temporary recall plaintext rows.
  /// 3. Refreshes active memories count notifier.
  Future<void> completeMemoryRetrieval(
    String retrievalId, {
    String? memoryId,
    String? reconstructedPlaintext,
    int packetCount = 0,
  }) async {
    final client = _supabase;
    if (client == null || _currentUser == null) return;

    try {
      if (memoryId != null &&
          reconstructedPlaintext != null &&
          reconstructedPlaintext.trim().isNotEmpty) {
        final entry = RecoveredMemory(
          memoryId: memoryId,
          reconstructedPlaintext: reconstructedPlaintext.trim(),
          recoveredAt: DateTime.now(),
          packetCount: packetCount,
        );
        await RecoveryHistoryService.instance.saveRecoveredMemory(entry);
        debugPrint(
          'RECOVERY HISTORY SAVED: Memory $memoryId plaintext stored in owner local history',
        );
      }

      await client.rpc(
        'complete_memory_retrieval',
        params: {'p_retrieval_id': retrievalId},
      );
      debugPrint(
        'RETRIEVAL COMPLETED: Server temporary recall plaintext buffer purged for $retrievalId',
      );

      await getActiveMemoriesCount(forceRefresh: true);
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

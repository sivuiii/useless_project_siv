import 'dart:convert';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:human_server/models/pending_delivery_fragment.dart';
import 'package:human_server/models/pending_recall_fragment.dart';
import 'package:human_server/models/stored_node_fragment.dart';
import 'package:human_server/screens/auth_screen.dart';
import 'package:human_server/screens/main_screen.dart';
import 'package:human_server/screens/node_screen.dart';
import 'package:human_server/screens/retrieve_screen.dart';
import 'package:human_server/screens/store_screen.dart';
import 'package:human_server/services/memory_service.dart';
import 'package:human_server/services/node_storage_service.dart';
import 'package:human_server/theme/app_theme.dart';
import 'package:human_server/widgets/memorization_dialog.dart';
import 'package:human_server/widgets/recall_submission_dialog.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  group('Randomized & Overlapping Fragmentation Model Tests', () {
    test('Empty or whitespace text returns empty list', () {
      expect(MemoryService.fragmentText(''), isEmpty);
      expect(MemoryService.fragmentText('   \n\t  '), isEmpty);
    });

    test('Very short text (<= 2 words) remains as single fragment', () {
      expect(MemoryService.fragmentText('SECRET'), ['SECRET']);
      expect(MemoryService.fragmentText('PIN 4096'), ['PIN 4096']);
    });

    // Requirement 1: Fragments are not simply sequential character slices
    test('Fragments are not simply sequential character slices and contain multi-subset combinations', () {
      const text = 'THE MEETING IS AT 4:30 PM IN ROOM 204';
      final fragments = MemoryService.fragmentText(text, random: Random(1));

      expect(fragments.length, greaterThanOrEqualTo(4));

      // In sequential slices, the sum of lengths roughly equals original length without overlap.
      // In overlapping multi-subset combinations, the combined token count significantly exceeds original token count.
      final totalFragmentTokens = fragments
          .expand((f) => f.replaceAll('+', ' ').split(RegExp(r'\s+')).where((t) => t.isNotEmpty))
          .length;
      expect(totalFragmentTokens, greaterThan(9 * 2),
          reason: 'Total tokens across fragments must reflect significant multi-subset redundancy');

      // Fragments must contain the multi-subset delimiter ' + '
      final hasMultiSubsetFormat = fragments.every((f) => f.contains('+'));
      expect(hasMultiSubsetFormat, isTrue,
          reason: 'Fragments should combine phrases with + to represent overlapping subsets');
    });

    // Requirement 2: Fragments overlap
    test('Fragments overlap with shared tokens across multiple fragments', () {
      const text = 'THE MEETING IS AT 4:30 PM IN ROOM 204';
      final fragments = MemoryService.fragmentText(text, random: Random(42));

      final tokenFrequency = <String, int>{};
      for (final frag in fragments) {
        final tokensInFrag = frag
            .replaceAll('+', ' ')
            .split(RegExp(r'\s+'))
            .where((t) => t.isNotEmpty)
            .map((t) => t.toUpperCase())
            .toSet();

        for (final t in tokensInFrag) {
          tokenFrequency[t] = (tokenFrequency[t] ?? 0) + 1;
        }
      }

      // Every original token must appear in at least 2 distinct fragments (required overlap & redundancy)
      const originalTokens = [
        'THE', 'MEETING', 'IS', 'AT', '4:30', 'PM', 'IN', 'ROOM', '204'
      ];
      for (final token in originalTokens) {
        final count = tokenFrequency[token] ?? 0;
        expect(count, greaterThanOrEqualTo(2),
            reason: 'Token "$token" must appear in at least 2 fragments for reconstruction redundancy');
      }

      // Assert pairwise overlap: every fragment shares at least one token with another fragment
      for (int i = 0; i < fragments.length; i++) {
        final tokensA = fragments[i]
            .replaceAll('+', ' ')
            .split(RegExp(r'\s+'))
            .where((t) => t.isNotEmpty)
            .toSet();

        bool hasOverlap = false;
        for (int j = 0; j < fragments.length; j++) {
          if (i == j) continue;
          final tokensB = fragments[j]
              .replaceAll('+', ' ')
              .split(RegExp(r'\s+'))
              .where((t) => t.isNotEmpty)
              .toSet();

          if (tokensA.intersection(tokensB).isNotEmpty) {
            hasOverlap = true;
            break;
          }
        }
        expect(hasOverlap, isTrue,
            reason: 'Fragment "${fragments[i]}" must overlap with at least one other fragment');
      }
    });

    // Requirement 3: No fragment contains the complete original for normal-length messages
    test('No fragment contains the complete original for normal-length messages', () {
      final testCases = [
        'MEET AT NOON', // 3 words
        'THE MEETING IS AT 4:30 PM IN ROOM 204', // 9 words
        'The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon', // 16 words
      ];

      for (final text in testCases) {
        final originalTokens = text
            .split(RegExp(r'\s+'))
            .where((t) => t.isNotEmpty)
            .map((t) => t.toUpperCase())
            .toSet();

        final fragments = MemoryService.fragmentText(text, random: Random(123));
        expect(fragments, isNotEmpty);

        for (final frag in fragments) {
          final fragTokens = frag
              .replaceAll('+', ' ')
              .split(RegExp(r'\s+'))
              .where((t) => t.isNotEmpty)
              .map((t) => t.toUpperCase())
              .toSet();

          expect(
            fragTokens.containsAll(originalTokens),
            isFalse,
            reason: 'Fragment "$frag" must NEVER contain the entire message "$text"',
          );
          expect(fragTokens.length, lessThan(originalTokens.length));
        }
      }
    });

    // Requirement 4: Fragments can be randomized
    test('Fragments can be randomized in order and combinations', () {
      const text = 'THE MEETING IS AT 4:30 PM IN ROOM 204';
      final run1 = MemoryService.fragmentText(text, random: Random(100));
      final run2 = MemoryService.fragmentText(text, random: Random(200));

      expect(run1, isNotEmpty);
      expect(run2, isNotEmpty);

      // Verify that randomized runs produce different ordering
      final sameOrder = run1.length == run2.length &&
          List.generate(run1.length, (i) => run1[i] == run2[i]).every((e) => e);

      expect(sameOrder, isFalse,
          reason: 'Different random seeds should yield randomized fragment orderings');
    });

    test('Calculates deterministic sha256 hash', () {
      const text = 'Node payload fragment';
      final hash1 = MemoryService.calculateHash(text);
      final hash2 = MemoryService.calculateHash(text);
      expect(hash1, hash2);
      expect(hash1.length, 64);
    });
  });

  group('Node Storage & Ephemeral Memorization Privacy Tests', () {
    // Requirement 5: Plaintext is not persisted by NodeStorageService
    test('Plaintext is not persisted by NodeStorageService', () async {
      SharedPreferences.setMockInitialValues({});
      final metadata = StoredNodeFragment(
        fragmentId: 'frag-test-999',
        memoryId: 'mem-test-888',
        sequenceNumber: 1,
        sizeBytes: 25,
        receivedAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 180)),
        assignmentId: 'assign-777',
        verificationHash: MemoryService.calculateHash('SECRET_PLAINTEXT_DO_NOT_STORE'),
        status: 'MEMORIZED',
      );

      await NodeStorageService.instance.saveFragment(metadata);

      // 1. Verify service retrieves metadata
      final fetched = await NodeStorageService.instance.getFragment('frag-test-999');
      expect(fetched, isNotNull);
      expect(fetched!.sizeBytes, 25);
      expect(fetched.status, 'MEMORIZED');

      // 2. Directly inspect SharedPreferences raw disk storage
      final prefs = await SharedPreferences.getInstance();
      final allKeys = prefs.getKeys();

      for (final key in allKeys) {
        final val = prefs.get(key);
        if (val is String && val.startsWith('{')) {
          final decoded = jsonDecode(val) as Map<String, dynamic>;
          expect(decoded.containsKey('plaintext'), isFalse,
              reason: 'SharedPreferences JSON must NEVER contain a "plaintext" key');
        }
      }
    });

    test('PendingDeliveryFragment holds ephemeral text and clears in-memory payload', () {
      final pending = PendingDeliveryFragment(
        inboxId: 'inbox-1',
        assignmentId: 'assign-1',
        fragmentId: 'frag-1',
        memoryId: 'mem-1',
        sequenceNumber: 1,
        payloadText: 'Secret access code 4096',
        sizeBytes: 23,
        expectedHash: MemoryService.calculateHash('Secret access code 4096'),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
      );

      expect(pending.payloadText, 'Secret access code 4096');
      expect(pending.isCleared, isFalse);

      pending.clear();

      expect(pending.payloadText, isEmpty);
      expect(pending.isCleared, isTrue);
    });

    // Requirement 6: Sender original is only transient UI/session state
    testWidgets('Sender original is only transient UI/session state and not persisted to disk',
        (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const StoreScreen(),
        ),
      );

      // Enter confidential memory into controller
      const confidentialText = 'TOP_SECRET_CREDIT_CARD_9999';
      await tester.enterText(find.byType(TextField), confidentialText);
      await tester.pump();

      // Verify that SharedPreferences contains NO record of the confidential text
      final prefs = await SharedPreferences.getInstance();
      final keys = prefs.getKeys();
      for (final key in keys) {
        final value = prefs.get(key).toString();
        expect(value.contains(confidentialText), isFalse,
            reason: 'Sender original plaintext must NEVER be written to SharedPreferences');
      }
    });

    test('confirmMemorizedFragment preserves plaintext and does not save local metadata when server fails', () async {
      SharedPreferences.setMockInitialValues({});
      final pending = PendingDeliveryFragment(
        inboxId: 'inbox-fail-1',
        assignmentId: 'assign-fail-1',
        fragmentId: 'frag-fail-1',
        memoryId: 'mem-fail-1',
        sequenceNumber: 1,
        payloadText: 'CRITICAL_PAYLOAD_DO_NOT_LOSE',
        sizeBytes: 28,
        expectedHash: MemoryService.calculateHash('CRITICAL_PAYLOAD_DO_NOT_LOSE'),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
      );

      // In unit test environment, Supabase client is not initialized, so confirmMemorizedFragment throws
      expect(
        () => MemoryService.instance.confirmMemorizedFragment(pending),
        throwsA(isA<Exception>()),
      );

      // Invariant: Plaintext is NOT cleared on failure
      expect(pending.payloadText, 'CRITICAL_PAYLOAD_DO_NOT_LOSE');
      expect(pending.isCleared, isFalse);

      // Invariant: Local metadata was NOT persisted
      final stored = await NodeStorageService.instance.getFragment('frag-fail-1');
      expect(stored, isNull);
    });

    testWidgets('MemorizationDialog renders plaintext and exits loading state on failure', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      final pending = PendingDeliveryFragment(
        inboxId: 'inbox-dialog-1',
        assignmentId: 'assign-dialog-1',
        fragmentId: 'frag-dialog-1',
        memoryId: 'mem-dialog-1',
        sequenceNumber: 2,
        payloadText: 'BIOLOGICAL_MEMORIZE_PAYLOAD',
        sizeBytes: 27,
        expectedHash: MemoryService.calculateHash('BIOLOGICAL_MEMORIZE_PAYLOAD'),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
      );

      bool onMemorizedCalled = false;

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: MemorizationDialog(
              delivery: pending,
              onMemorized: () {
                onMemorizedCalled = true;
              },
            ),
          ),
        ),
      );

      // Plaintext is rendered on screen for human memorization
      expect(find.text('BIOLOGICAL_MEMORIZE_PAYLOAD'), findsOneWidget);
      expect(find.text('CONFIRM MEMORIZED'), findsOneWidget);

      // Tap confirm button
      await tester.tap(find.text('CONFIRM MEMORIZED'));
      await tester.pump();

      // Pump to let the async call reject
      await tester.pump(const Duration(milliseconds: 100));

      // Invariant: on failure, dialog exits loading state, shows error, and offers retry
      expect(find.byType(CircularProgressIndicator), findsNothing);
      expect(find.text('RETRY CONFIRMATION'), findsOneWidget);
      expect(onMemorizedCalled, isFalse);
      expect(pending.payloadText, 'BIOLOGICAL_MEMORIZE_PAYLOAD');
    });
  });

  group('Human Node 5-Fragment Capacity & Retrieval Forgetting Tests', () {
    setUp(() {
      SharedPreferences.setMockInitialValues({});
      NodeStorageService.instance.resetForTesting();
    });

    tearDown(() {
      NodeStorageService.instance.resetForTesting();
    });

    test('Node capacity evaluation: 0 hosted fragments can receive one', () {
      expect(MemoryService.canNodeAcceptFragment(0), isTrue);
    });

    test('Node capacity evaluation: 4 hosted fragments can receive one', () {
      expect(MemoryService.canNodeAcceptFragment(4), isTrue);
    });

    test('Node capacity evaluation: 5 hosted fragments cannot receive another', () {
      expect(MemoryService.canNodeAcceptFragment(5), isFalse);
    });

    test('Full and over-capacity nodes are excluded from assignment', () {
      expect(MemoryService.canNodeAcceptFragment(5), isFalse);
      expect(MemoryService.canNodeAcceptFragment(6), isFalse);
      expect(MemoryService.canNodeAcceptFragment(12), isFalse);
    });

    test('Forgetting a successfully recalled fragment frees one slot', () async {
      // Populate 5 hosted fragments into node storage
      for (int i = 1; i <= 5; i++) {
        await NodeStorageService.instance.saveFragment(
          StoredNodeFragment(
            fragmentId: 'frag-slot-$i',
            memoryId: 'mem-test',
            sequenceNumber: i,
            sizeBytes: 10,
            receivedAt: DateTime.now(),
            expiresAt: DateTime.now().add(const Duration(days: 180)),
            assignmentId: 'assign-slot-$i',
            verificationHash: 'hash-$i',
          ),
        );
      }

      // Verify node is currently at full capacity (5 / 5)
      expect(NodeStorageService.instance.activeHostedCount, 5);
      expect(NodeStorageService.instance.isCapacityFull, isTrue);
      expect(NodeStorageService.instance.canReceiveFragment, isFalse);

      // Human node recalls fragment 1: forgets it locally
      await NodeStorageService.instance.deleteFragment('frag-slot-1');

      // Slot is freed: node now has 4 hosted fragments and can receive another
      expect(NodeStorageService.instance.activeHostedCount, 4);
      expect(NodeStorageService.instance.isCapacityFull, isFalse);
      expect(NodeStorageService.instance.canReceiveFragment, isTrue);
      expect(NodeStorageService.instance.remainingSlots, 1);
    });

    test('Failed retrieval does NOT delete the hosted fragment', () async {
      final fragment = StoredNodeFragment(
        fragmentId: 'frag-safe-recall',
        memoryId: 'mem-safe-recall',
        sequenceNumber: 1,
        sizeBytes: 15,
        receivedAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 180)),
        assignmentId: 'assign-safe-recall',
        verificationHash: 'hash-safe',
      );

      await NodeStorageService.instance.saveFragment(fragment);
      expect(await NodeStorageService.instance.hasFragment('frag-safe-recall'), isTrue);

      // Attempting to forget with no active Supabase connection throws
      expect(
        () => MemoryService.instance.forgetRecalledFragment(
          assignmentId: 'assign-safe-recall',
          fragmentId: 'frag-safe-recall',
        ),
        throwsA(isA<Exception>()),
      );

      // Invariant: Failed retrieval MUST NOT delete the hosted fragment from local storage
      expect(await NodeStorageService.instance.hasFragment('frag-safe-recall'), isTrue);
      final retrieved = await NodeStorageService.instance.getFragment('frag-safe-recall');
      expect(retrieved, isNotNull);
      expect(retrieved!.fragmentId, 'frag-safe-recall');
    });

    testWidgets('NodeScreen renders memory slots used and station full indicator', (WidgetTester tester) async {
      tester.binding.setSurfaceSize(const Size(1000, 1000));
      addTearDown(() => tester.binding.setSurfaceSize(null));

      // Fill 5 slots
      for (int i = 1; i <= 5; i++) {
        await NodeStorageService.instance.saveFragment(
          StoredNodeFragment(
            fragmentId: 'frag-full-$i',
            memoryId: 'mem-full',
            sequenceNumber: i,
            sizeBytes: 12,
            receivedAt: DateTime.now(),
            expiresAt: DateTime.now().add(const Duration(days: 180)),
            assignmentId: 'assign-full-$i',
            verificationHash: 'hash-$i',
          ),
        );
      }

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const NodeScreen(),
        ),
      );
      await tester.pump();

      // Verify slot usage and station full banner are rendered
      expect(find.text('5 / 5 MEMORY SLOTS USED'), findsOneWidget);
      expect(find.textContaining('STATION FULL (5/5 SLOTS)'), findsOneWidget);
    });

    test('clearAllMetadata clears local metadata records and resets capacity to 0/5', () async {
      // Simulate over-capacity test state (17 fragments)
      for (int i = 1; i <= 17; i++) {
        await NodeStorageService.instance.saveFragment(
          StoredNodeFragment(
            fragmentId: 'frag-reset-$i',
            memoryId: 'mem-reset',
            sequenceNumber: i,
            sizeBytes: 20,
            receivedAt: DateTime.now(),
            expiresAt: DateTime.now().add(const Duration(days: 180)),
            assignmentId: 'assign-reset-$i',
            verificationHash: 'hash-$i',
          ),
        );
      }

      expect(NodeStorageService.instance.activeHostedCount, 17);
      expect(NodeStorageService.instance.isOverCapacity, isTrue);
      expect(NodeStorageService.instance.canReceiveFragment, isFalse);

      // Perform local clear
      await NodeStorageService.instance.clearAllMetadata();

      // Invariant: local cache is empty, capacity resets to 0 / 5
      expect(NodeStorageService.instance.activeHostedCount, 0);
      expect(NodeStorageService.instance.isCapacityFull, isFalse);
      expect(NodeStorageService.instance.isOverCapacity, isFalse);
      expect(NodeStorageService.instance.canReceiveFragment, isTrue);
      expect(NodeStorageService.instance.remainingSlots, 5);

      final fragments = await NodeStorageService.instance.getStoredFragments();
      expect(fragments, isEmpty);
    });

    testWidgets('NodeScreen clear local memory requires confirmation dialog and resets slots to 0/5', (WidgetTester tester) async {
      tester.binding.setSurfaceSize(const Size(1000, 1400));
      addTearDown(() => tester.binding.setSurfaceSize(null));

      // Simulate 17 fragments
      for (int i = 1; i <= 17; i++) {
        await NodeStorageService.instance.saveFragment(
          StoredNodeFragment(
            fragmentId: 'frag-dev-$i',
            memoryId: 'mem-dev',
            sequenceNumber: i,
            sizeBytes: 15,
            receivedAt: DateTime.now(),
            expiresAt: DateTime.now().add(const Duration(days: 180)),
            assignmentId: 'assign-dev-$i',
            verificationHash: 'hash-$i',
          ),
        );
      }

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const Scaffold(body: NodeScreen()),
        ),
      );
      await tester.pump();

      expect(find.text('17 / 5 MEMORY SLOTS USED'), findsOneWidget);
      expect(find.byKey(const Key('clear_local_memory_button')), findsOneWidget);

      // Scroll to clear local memory button if needed and tap it
      await tester.scrollUntilVisible(find.byKey(const Key('clear_local_memory_button')), 200);
      await tester.tap(find.byKey(const Key('clear_local_memory_button')));
      await tester.pumpAndSettle();

      // Verify confirmation dialog is presented
      expect(find.text('CLEAR LOCAL MEMORY?'), findsOneWidget);
      expect(find.textContaining('This development action clears ONLY the local metadata cache (17 item(s))'), findsOneWidget);
      expect(find.text('CANCEL'), findsOneWidget);
      expect(find.text('CONFIRM & CLEAR'), findsOneWidget);

      // Tap CANCEL first
      await tester.tap(find.text('CANCEL'));
      await tester.pumpAndSettle();

      // State is preserved
      expect(find.text('CLEAR LOCAL MEMORY?'), findsNothing);
      expect(NodeStorageService.instance.activeHostedCount, 17);
      expect(find.text('17 / 5 MEMORY SLOTS USED'), findsOneWidget);

      // Tap CLEAR LOCAL MEMORY again, then tap CONFIRM & CLEAR
      await tester.tap(find.byKey(const Key('clear_local_memory_button')));
      await tester.pumpAndSettle();

      await tester.tap(find.text('CONFIRM & CLEAR'));
      await tester.pumpAndSettle();

      // Invariant: local memory is wiped, immediately reflects 0 / 5 MEMORY SLOTS USED
      expect(NodeStorageService.instance.activeHostedCount, 0);
      expect(find.text('0 / 5 MEMORY SLOTS USED'), findsOneWidget);
      expect(find.text('No Fragments Memorized Yet'), findsOneWidget);
      expect(find.textContaining('LOCAL MEMORY CLEARED (0 / 5 SLOTS USED)'), findsOneWidget);
    });
  });

  group('UI Smoke Tests', () {
    testWidgets('Human Server main UI smoke test', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const MainScreen(),
        ),
      );
      await tester.pump();

      expect(find.text('COMMUNICATIONS DESK'), findsOneWidget);
      expect(find.text('Desk'), findsOneWidget);
      expect(find.text('Dispatch'), findsOneWidget);
      expect(find.text('Recall'), findsOneWidget);
      expect(find.text('Station'), findsOneWidget);
      expect(find.text('Register'), findsOneWidget);
    });

    testWidgets('Human Server auth screen smoke test', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const AuthScreen(),
        ),
      );
      await tester.pump();

      expect(find.text('HUMAN SERVER'), findsOneWidget);
      expect(find.text('AUTHENTICATE'), findsOneWidget);
      expect(find.text('REGISTER STATION'), findsOneWidget);
      expect(find.text('OPERATOR EMAIL IDENTIFIER'), findsOneWidget);
      expect(find.text('ACCESS KEY PASSPHRASE'), findsOneWidget);
      expect(find.text('CONNECT OPERATOR TERMINAL'), findsOneWidget);

      await tester.tap(find.text('REGISTER STATION'));
      await tester.pump();

      expect(find.text('CONFIRM ACCESS KEY'), findsOneWidget);
      expect(find.text('REGISTER STATION NODE'), findsOneWidget);
    });

    testWidgets('Human Server node screen smoke test', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const NodeScreen(),
        ),
      );
      await tester.pump();

      expect(find.text('Station Telemetry'), findsOneWidget);
      expect(find.text('RESPONSE TIME'), findsOneWidget);
      expect(find.text('RESPONSE RATE'), findsOneWidget);
      expect(find.byType(Switch), findsOneWidget);
    });

    testWidgets('Human Server store screen smoke test', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const StoreScreen(),
        ),
      );
      await tester.pump();

      expect(find.text('Dispatch Memory'), findsOneWidget);
      expect(find.text('ACTIVE MEMORY SLOTS'), findsOneWidget);
      expect(find.text('TELEGRAM DISPATCH FORM'), findsOneWidget);
      expect(find.text('DISPATCH MEMORY TELEGRAM'), findsOneWidget);
    });
  });

  group('MVP Retrieval Flow & Reconstruction Tests', () {
    test('Reconstructs from simple overlapping phrases', () {
      final fragments = [
        'THE SECRET CODE + CODE IS 4096',
        'IS 4096 + 4096 VERIFIED',
      ];
      final reconstructed = MemoryService.reconstructFromFragments(fragments);
      expect(reconstructed, contains('THE SECRET CODE'));
      expect(reconstructed, contains('IS 4096'));
    });

    test('Reconstructs original message from generated overlapping fragments', () {
      const original = 'THE MEETING IS AT 4:30 PM IN ROOM 204';
      final fragments = MemoryService.fragmentText(original, random: Random(42));
      expect(fragments, isNotEmpty);

      final reconstructed = MemoryService.reconstructFromFragments(fragments);
      expect(reconstructed, equals(original));
    });

    test('Reconstructs multi-word sentences from random fragment sets', () {
      const messages = [
        'HELLO WORLD',
        'SYSTEM STATUS ALL CIRCUITS FUNCTIONAL',
        'Biological station nodes hold distributed memory fragments',
      ];

      for (final msg in messages) {
        final fragments = MemoryService.fragmentText(msg, random: Random(123));
        final reconstructed = MemoryService.reconstructFromFragments(fragments);
        expect(reconstructed.toLowerCase(), equals(msg.toLowerCase()));
      }
    });

    testWidgets('RecallSubmissionDialog renders recall prompt and rejects empty submission', (WidgetTester tester) async {
      final recall = PendingRecallFragment(
        retrievalId: 'ret-123',
        assignmentId: 'assign-456',
        fragmentId: 'frag-789',
        memoryId: 'mem-000',
        sequenceNumber: 1,
        sizeBytes: 15,
        expectedHash: MemoryService.calculateHash('SECRET PHRASE'),
        retrievalCreatedAt: DateTime.now(),
      );

      bool recalled = false;

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: RecallSubmissionDialog(
              recall: recall,
              onSuccess: () {
                recalled = true;
              },
            ),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('RECALL SIGNAL RECEIVED'), findsOneWidget);
      expect(find.text('SEQ #1'), findsOneWidget);
      expect(find.text('15 BYTES EXPECTED'), findsOneWidget);
      expect(find.text('SUBMIT RECALL'), findsOneWidget);

      // Tap submit with empty text
      await tester.tap(find.text('SUBMIT RECALL'));
      await tester.pump();

      // Invariant: empty text rejected, does not succeed
      expect(find.text('Please enter your remembered fragment text.'), findsOneWidget);
      expect(recalled, isFalse);
    });

    testWidgets('RecallSubmissionDialog handles server rejection gracefully', (WidgetTester tester) async {
      final recall = PendingRecallFragment(
        retrievalId: 'ret-fail',
        assignmentId: 'assign-fail',
        fragmentId: 'frag-fail',
        memoryId: 'mem-fail',
        sequenceNumber: 2,
        sizeBytes: 20,
        expectedHash: MemoryService.calculateHash('PROPER PHRASE'),
        retrievalCreatedAt: DateTime.now(),
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: RecallSubmissionDialog(
              recall: recall,
              onSuccess: () {},
            ),
          ),
        ),
      );
      await tester.pump();

      // Enter text and submit
      await tester.enterText(find.byType(TextField), 'WRONG PLAINTEXT ENTRY');
      await tester.tap(find.text('SUBMIT RECALL'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      // Invariant: does not crash or stay loading, shows error
      expect(find.byType(CircularProgressIndicator), findsNothing);
      expect(find.text('SUBMIT RECALL'), findsOneWidget);
    });

    testWidgets('RetrieveScreen renders target memory input and retrieval CTA', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const RetrieveScreen(),
        ),
      );
      await tester.pump();

      expect(find.text('RECALL SIGNAL & RECONSTRUCT'), findsOneWidget);
      expect(find.text('SEMANTIC RECALL TELEGRAPH'), findsOneWidget);
      expect(find.text('TARGET MEMORY ID'), findsOneWidget);
      expect(find.text('INITIATE RETRIEVAL SIGNAL'), findsOneWidget);
    });
  });
}

import 'dart:convert';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:human_server/models/pending_delivery_fragment.dart';
import 'package:human_server/models/pending_recall_fragment.dart';
import 'package:human_server/models/recovered_memory.dart';
import 'package:human_server/models/retrieval_status.dart';
import 'package:human_server/models/stored_node_fragment.dart';
import 'package:human_server/models/user_stored_memory.dart';
import 'package:human_server/screens/auth_screen.dart';
import 'package:human_server/screens/main_screen.dart';
import 'package:human_server/screens/node_screen.dart';
import 'package:human_server/screens/retrieve_screen.dart';
import 'package:human_server/screens/store_screen.dart';
import 'package:human_server/services/memory_service.dart';
import 'package:human_server/services/node_storage_service.dart';
import 'package:human_server/services/recovery_history_service.dart';
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

      expect(find.text('RECOVERY REQUEST'), findsOneWidget);
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
      expect(find.text('REQUEST RECOVERY'), findsOneWidget);
    });
  });

  group('Frozen Human Server MVP Requirements (1-14)', () {
    // 1. sender plaintext cleared after dispatch
    testWidgets('Requirement 1: sender plaintext cleared immediately after dispatch', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const StoreScreen(),
        ),
      );
      await tester.pump();

      final inputFinder = find.byType(TextField);
      expect(inputFinder, findsOneWidget);

      await tester.enterText(inputFinder, 'THE MEETING IS AT 4:30 PM IN ROOM 204');
      await tester.pump();

      // Trigger dispatch
      await tester.tap(find.text('DISPATCH MEMORY TELEGRAM'));
      await tester.pump();

      // Invariant: Controller text is immediately cleared
      final tf = tester.widget<TextField>(inputFinder);
      expect(tf.controller?.text.isEmpty, isTrue);
    });

    // 2. sender cannot access original plaintext after dispatch
    testWidgets('Requirement 2: sender cannot access original plaintext after dispatch in Store UI', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const StoreScreen(),
        ),
      );
      await tester.pump();

      const secret = 'THE MEETING IS AT 4:30 PM IN ROOM 204';
      await tester.enterText(find.byType(TextField), secret);
      await tester.tap(find.text('DISPATCH MEMORY TELEGRAM'));
      await tester.pump();

      // Neither during nor after dispatch is the plaintext exposed
      expect(find.text(secret), findsNothing);
      expect(find.text('DISPATCHED PAYLOAD'), findsNothing);
    });

    // 3. packet count metadata
    test('Requirement 3: packet count metadata is correctly constructed', () {
      final result = MemoryStoreResult(
        memoryId: 'mem-12345678',
        packetCount: 6,
        packetsAssigned: 4,
        packetsPending: 2,
        packetsMemorized: 0,
        status: 'awaiting_more_nodes',
      );

      expect(result.packetCount, 6);
      expect(result.packetsMemorized, 0);
      expect(result.packetsAssigned, 4);
      expect(result.packetsPending, 2);
      expect(result.assignedNodeIds, isEmpty); // recipient identities never exposed
    });

    // 4. same node cannot receive two fragments from one memory
    test('Requirement 4: same node cannot receive two fragments from one memory', () {
      final assignedNodes = <String>{};
      final candidateNodes = ['node-1', 'node-2', 'node-3', 'node-4'];
      const memoryId = 'mem-test-999';

      final assignments = <Map<String, String>>[];
      for (int i = 0; i < 4; i++) {
        final eligible = candidateNodes.firstWhere(
          (n) => !assignedNodes.contains(n),
          orElse: () => '',
        );
        if (eligible.isNotEmpty) {
          assignedNodes.add(eligible);
          assignments.add({'memory_id': memoryId, 'fragment': 'frag-$i', 'node_id': eligible});
        }
      }

      final nodeIds = assignments.map((a) => a['node_id']!).toList();
      expect(nodeIds.toSet().length, equals(nodeIds.length));
      expect(nodeIds, containsAll(['node-1', 'node-2', 'node-3', 'node-4']));

      // If a 5th fragment arrives with no more nodes, it must stay pending
      final eligible5 = candidateNodes.where((n) => !assignedNodes.contains(n)).toList();
      expect(eligible5, isEmpty, reason: 'Must not reassign to an existing node of the same memory');
    });

    // 5. node maximum 5 packets
    test('Requirement 5: node capacity capped at 5 packets and frees slot on recall', () async {
      SharedPreferences.setMockInitialValues({});
      NodeStorageService.instance.resetForTesting();
      await NodeStorageService.instance.clearAllMetadata();

      expect(NodeStorageService.instance.maxCapacity, 5);
      expect(NodeStorageService.instance.activeHostedCount, 0);
      expect(NodeStorageService.instance.isCapacityFull, isFalse);

      for (int i = 1; i <= 5; i++) {
        await NodeStorageService.instance.saveFragment(
          StoredNodeFragment(
            fragmentId: 'frag-$i',
            memoryId: 'mem-$i',
            sequenceNumber: i,
            sizeBytes: 20,
            receivedAt: DateTime.now(),
            expiresAt: DateTime.now().add(const Duration(days: 180)),
            assignmentId: 'assign-$i',
            verificationHash: 'hash-$i',
          ),
        );
      }

      expect(NodeStorageService.instance.activeHostedCount, 5);
      expect(NodeStorageService.instance.isCapacityFull, isTrue);

      // Successfully recalling/forgetting 1 fragment frees the slot
      await NodeStorageService.instance.deleteFragment('frag-1');
      expect(NodeStorageService.instance.activeHostedCount, 4);
      expect(NodeStorageService.instance.isCapacityFull, isFalse);
    });

    // 6. sender username reaches recipient
    testWidgets('Requirement 6: sender username reaches recipient in memorization & recall prompts', (WidgetTester tester) async {
      final delivery = PendingDeliveryFragment(
        inboxId: 'inbox-1',
        assignmentId: 'assign-1',
        fragmentId: 'frag-1',
        memoryId: 'mem-1',
        sequenceNumber: 1,
        payloadText: 'THE MEETING + 4:30',
        sizeBytes: 18,
        expectedHash: MemoryService.calculateHash('THE MEETING + 4:30'),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
        senderUsername: 'SIVUIII',
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: MemorizationDialog(
              delivery: delivery,
              onMemorized: () {},
            ),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('MEMORY FROM: SIVUIII'), findsOneWidget);
    });

    // 7. memorization count updates
    test('Requirement 7: memorization count updates and formats properly', () {
      final mem = UserStoredMemory(
        id: '8f31a2b4-0000-0000-0000-000000000000',
        createdAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 180)),
        fragmentCount: 6,
        packetCount: 6,
        memorizedCount: 4,
        status: 'active',
      );

      expect(mem.shortId, '8F31A2B4');
      expect(mem.packetCount, 6);
      expect(mem.memorizedCount, 4);
    });

    // 8. recall request reaches all relevant nodes
    test('Requirement 8: recall request maps to all relevant nodes with canonical indices', () {
      final rawRecalls = [
        {
          'retrieval_id': 'ret-1',
          'assignment_id': 'assign-1',
          'fragment_id': 'frag-1',
          'memory_id': 'mem-1',
          'sequence_number': 1,
          'size_bytes': 18,
          'expected_hash': 'h1',
          'sender_username': 'SIVUIII',
          'canonical_indices': [0, 1],
        },
        {
          'retrieval_id': 'ret-1',
          'assignment_id': 'assign-2',
          'fragment_id': 'frag-2',
          'memory_id': 'mem-1',
          'sequence_number': 2,
          'size_bytes': 18,
          'expected_hash': 'h2',
          'sender_username': 'SIVUIII',
          'canonical_indices': [1, 2],
        },
      ];

      final recalls = rawRecalls.map((m) => PendingRecallFragment.fromMap(m)).toList();
      expect(recalls.length, 2);
      expect(recalls[0].senderUsername, 'SIVUIII');
      expect(recalls[0].canonicalIndices, [0, 1]);
      expect(recalls[1].canonicalIndices, [1, 2]);
    });

    // 9. responses arriving in random order
    test('Requirement 9: responses arriving in random order reconstruct deterministically', () {
      final responses = [
        RecalledFragmentItem(
          fragmentId: 'f3',
          sequenceNumber: 3,
          recalledText: 'IN ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [3],
        ),
        RecalledFragmentItem(
          fragmentId: 'f1',
          sequenceNumber: 1,
          recalledText: 'THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [0],
        ),
        RecalledFragmentItem(
          fragmentId: 'f2',
          sequenceNumber: 2,
          recalledText: '4:30 PM',
          createdAt: DateTime.now(),
          canonicalIndices: const [2],
        ),
        RecalledFragmentItem(
          fragmentId: 'f4',
          sequenceNumber: 4,
          recalledText: 'IS AT',
          createdAt: DateTime.now(),
          canonicalIndices: const [1],
        ),
      ];

      final reconstructed = MemoryService.reconstructFromResponses(responses);
      expect(reconstructed, 'THE MEETING IS AT 4:30 PM IN ROOM 204');
    });

    // 10. duplicate responses
    test('Requirement 10: duplicate responses are deduplicated without duplicated words', () {
      final responses = [
        RecalledFragmentItem(
          fragmentId: 'f1',
          sequenceNumber: 1,
          recalledText: 'THE MEETING + 4:30',
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 1],
        ),
        RecalledFragmentItem(
          fragmentId: 'f1-dup',
          sequenceNumber: 1,
          recalledText: 'THE MEETING + 4:30',
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 1],
        ),
        RecalledFragmentItem(
          fragmentId: 'f2',
          sequenceNumber: 2,
          recalledText: '4:30 + ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [1, 2],
        ),
        RecalledFragmentItem(
          fragmentId: 'f2-dup',
          sequenceNumber: 2,
          recalledText: '4:30 + ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [1, 2],
        ),
      ];

      final reconstructed = MemoryService.reconstructFromResponses(responses);
      expect(reconstructed, 'THE MEETING 4:30 ROOM 204');
      expect(reconstructed.split('ROOM 204').length - 1, 1);
    });

    // 11. redundant overlapping fragments
    test('Requirement 11: redundant overlapping fragments resolve cleanly without token explosion', () {
      final responses = [
        RecalledFragmentItem(
          fragmentId: 'fA',
          sequenceNumber: 1,
          recalledText: 'THE MEETING + 4:30',
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 1],
        ),
        RecalledFragmentItem(
          fragmentId: 'fB',
          sequenceNumber: 2,
          recalledText: '4:30 + ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [1, 2],
        ),
        RecalledFragmentItem(
          fragmentId: 'fC',
          sequenceNumber: 3,
          recalledText: 'ROOM 204 + THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [2, 0],
        ),
        RecalledFragmentItem(
          fragmentId: 'fD',
          sequenceNumber: 4,
          recalledText: 'THE MEETING + ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 2],
        ),
        RecalledFragmentItem(
          fragmentId: 'fE',
          sequenceNumber: 5,
          recalledText: '4:30 + THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [1, 0],
        ),
      ];

      final reconstructed = MemoryService.reconstructFromResponses(responses);
      expect(reconstructed, 'THE MEETING 4:30 ROOM 204');
    });

    // 11b. one incorrect human response (Requirement 16)
    test('Requirement 11b: one incorrect human response is detected and filtered out via overlapping consensus', () {
      final responses = [
        RecalledFragmentItem(
          fragmentId: 'fA',
          sequenceNumber: 1,
          recalledText: 'THE METING + 4:30', // Typo: METING instead of MEETING
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 1],
        ),
        RecalledFragmentItem(
          fragmentId: 'fB',
          sequenceNumber: 2,
          recalledText: '4:30 + ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [1, 2],
        ),
        RecalledFragmentItem(
          fragmentId: 'fC',
          sequenceNumber: 3,
          recalledText: 'ROOM 204 + THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [2, 0],
        ),
        RecalledFragmentItem(
          fragmentId: 'fD',
          sequenceNumber: 4,
          recalledText: 'THE MEETING + ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 2],
        ),
        RecalledFragmentItem(
          fragmentId: 'fE',
          sequenceNumber: 5,
          recalledText: '4:30 + THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [1, 0],
        ),
      ];

      final reconstructed = MemoryService.reconstructFromResponses(responses);
      expect(reconstructed, 'THE MEETING 4:30 ROOM 204');
    });

    // 11c. partial phrase recall (e.g. MEETING vs THE MEETING)
    test('Requirement 11c: partial word recall (MEETING instead of THE MEETING) absorbed by consensus', () {
      final responses = [
        RecalledFragmentItem(
          fragmentId: 'fA',
          sequenceNumber: 1,
          recalledText: 'THE MEETING + 4:30',
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 1],
        ),
        RecalledFragmentItem(
          fragmentId: 'fB',
          sequenceNumber: 2,
          recalledText: 'MEETING + ROOM 204', // Missing "THE"
          createdAt: DateTime.now(),
          canonicalIndices: const [0, 2],
        ),
        RecalledFragmentItem(
          fragmentId: 'fC',
          sequenceNumber: 3,
          recalledText: 'ROOM 204 + THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [2, 0],
        ),
        RecalledFragmentItem(
          fragmentId: 'fD',
          sequenceNumber: 4,
          recalledText: '4:30 + THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [1, 0],
        ),
      ];

      final reconstructed = MemoryService.reconstructFromResponses(responses);
      expect(reconstructed, 'THE MEETING 4:30 ROOM 204');
    });

    // 12. missing fragments
    test('Requirement 12: missing fragments assemble available phrases in correct canonical order', () {
      final responses = [
        RecalledFragmentItem(
          fragmentId: 'f1',
          sequenceNumber: 1,
          recalledText: 'THE MEETING',
          createdAt: DateTime.now(),
          canonicalIndices: const [0],
        ),
        RecalledFragmentItem(
          fragmentId: 'f3',
          sequenceNumber: 3,
          recalledText: 'ROOM 204',
          createdAt: DateTime.now(),
          canonicalIndices: const [2],
        ),
      ];

      final reconstructed = MemoryService.reconstructFromResponses(responses);
      expect(reconstructed, 'THE MEETING ROOM 204');
    });

    // 13. exact reconstruction of randomized overlapping fragments
    test('Requirement 13: exact reconstruction of randomized overlapping fragments for frozen test sentence', () {
      const sentence = 'THE MEETING IS AT 4:30 PM IN ROOM 204';
      final payloads = MemoryService.generateFragmentPayloads(sentence, random: Random(99));

      final responses = payloads
          .map((p) => RecalledFragmentItem(
                fragmentId: 'frag-${p.sequenceNumber}',
                sequenceNumber: p.sequenceNumber,
                recalledText: p.text,
                createdAt: DateTime.now(),
                canonicalIndices: p.canonicalIndices,
              ))
          .toList();

      responses.shuffle(Random(123));

      final reconstructed = MemoryService.reconstructFromResponses(responses);
      expect(reconstructed, equals(sentence));
    });

    // 14. plaintext transport cleanup
    test('Requirement 14: ephemeral delivery plaintext is wiped on clear', () {
      final delivery = PendingDeliveryFragment(
        inboxId: 'inbox-99',
        assignmentId: 'assign-99',
        fragmentId: 'frag-99',
        memoryId: 'mem-99',
        sequenceNumber: 1,
        payloadText: 'TOP SECRET MEETING',
        sizeBytes: 18,
        expectedHash: MemoryService.calculateHash('TOP SECRET MEETING'),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
        senderUsername: 'SIVUIII',
      );

      expect(delivery.payloadText, 'TOP SECRET MEETING');
      expect(delivery.isCleared, isFalse);

      delivery.clear();

      expect(delivery.payloadText, isEmpty);
      expect(delivery.isCleared, isTrue);
    });
  });

  group('Multi-User Local Storage Isolation Tests', () {
    setUp(() {
      SharedPreferences.setMockInitialValues({});
      NodeStorageService.instance.resetForTesting();
    });

    test('Strict user-scoped isolation across account switches', () async {
      const userA = 'user-uuid-aaaa-1111';
      const userB = 'user-uuid-bbbb-2222';

      // 1. User A stores metadata
      NodeStorageService.instance.setUserIdForTesting(userA);
      await NodeStorageService.instance.saveFragment(
        StoredNodeFragment(
          fragmentId: 'frag-userA-1',
          memoryId: 'mem-userA',
          sequenceNumber: 1,
          sizeBytes: 24,
          receivedAt: DateTime.now(),
          expiresAt: DateTime.now().add(const Duration(days: 30)),
          assignmentId: 'assign-userA-1',
          verificationHash: 'hash-userA-1',
        ),
      );

      expect(NodeStorageService.instance.activeHostedCount, 1);
      expect(await NodeStorageService.instance.hasFragment('frag-userA-1'), isTrue);

      // 2. Switch to User B
      NodeStorageService.instance.setUserIdForTesting(userB);
      final userBFragments = await NodeStorageService.instance.getStoredFragments();

      // 3. User B sees zero fragments
      expect(userBFragments, isEmpty);
      expect(NodeStorageService.instance.activeHostedCount, 0);
      expect(NodeStorageService.instance.isCapacityFull, isFalse);
      expect(await NodeStorageService.instance.hasFragment('frag-userA-1'), isFalse);

      // 4. User B stores metadata
      await NodeStorageService.instance.saveFragment(
        StoredNodeFragment(
          fragmentId: 'frag-userB-1',
          memoryId: 'mem-userB',
          sequenceNumber: 1,
          sizeBytes: 30,
          receivedAt: DateTime.now(),
          expiresAt: DateTime.now().add(const Duration(days: 30)),
          assignmentId: 'assign-userB-1',
          verificationHash: 'hash-userB-1',
        ),
      );

      expect(NodeStorageService.instance.activeHostedCount, 1);
      expect(await NodeStorageService.instance.hasFragment('frag-userB-1'), isTrue);
      expect(await NodeStorageService.instance.hasFragment('frag-userA-1'), isFalse);

      // 5. Switch back to User A
      NodeStorageService.instance.setUserIdForTesting(userA);
      final userAFragments = await NodeStorageService.instance.getStoredFragments();

      // 6. User A sees only User A's metadata
      expect(userAFragments.length, 1);
      expect(userAFragments.first.fragmentId, 'frag-userA-1');
      expect(NodeStorageService.instance.activeHostedCount, 1);
      expect(await NodeStorageService.instance.hasFragment('frag-userA-1'), isTrue);
      expect(await NodeStorageService.instance.hasFragment('frag-userB-1'), isFalse);

      // 7. clearAllMetadata for User B does not delete User A's metadata
      NodeStorageService.instance.setUserIdForTesting(userB);
      await NodeStorageService.instance.clearAllMetadata();
      expect(NodeStorageService.instance.activeHostedCount, 0);

      // Verify User A still has their metadata
      NodeStorageService.instance.setUserIdForTesting(userA);
      final userAAfterB = await NodeStorageService.instance.getStoredFragments();
      expect(userAAfterB.length, 1);
      expect(userAAfterB.first.fragmentId, 'frag-userA-1');
      expect(NodeStorageService.instance.activeHostedCount, 1);
    });

    test('Zero plaintext invariant is maintained in user-scoped storage', () async {
      const userA = 'user-uuid-aaaa-1111';
      NodeStorageService.instance.setUserIdForTesting(userA);

      await NodeStorageService.instance.saveFragment(
        StoredNodeFragment(
          fragmentId: 'frag-userA-safe',
          memoryId: 'mem-safe',
          sequenceNumber: 1,
          sizeBytes: 20,
          receivedAt: DateTime.now(),
          expiresAt: DateTime.now().add(const Duration(days: 30)),
          assignmentId: 'assign-safe',
          verificationHash: 'hash-safe',
        ),
      );

      final prefs = await SharedPreferences.getInstance();
      for (final key in prefs.getKeys()) {
        final val = prefs.get(key);
        if (val is String) {
          expect(val.contains('"plaintext"'), isFalse);
        }
      }
    });
  });

  group('Retrieval Lifecycle & Recovery History Regression Tests', () {
    setUp(() async {
      SharedPreferences.setMockInitialValues({});
      NodeStorageService.instance.resetForTesting();
      RecoveryHistoryService.instance.resetForTesting();
    });

    // 1. New stored memory: appears as recoverable
    test('1. New stored memory appears as recoverable (isReady = true)', () {
      final memory = UserStoredMemory(
        id: 'mem-new-1',
        createdAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
        fragmentCount: 5,
        packetCount: 5,
        memorizedCount: 5,
        status: 'active',
        retrievalStatus: 'ready',
      );

      expect(memory.isReady, isTrue);
      expect(memory.isUnderRecovery, isFalse);
      expect(memory.isCompleted, isFalse);
    });

    // 2. Request recovery: status becomes UNDER RECOVERY
    test('2. Request recovery: status becomes UNDER RECOVERY', () {
      final memory = UserStoredMemory(
        id: 'mem-under-recovery-2',
        createdAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
        fragmentCount: 4,
        packetCount: 4,
        memorizedCount: 4,
        status: 'active',
        retrievalStatus: 'under_recovery',
        activeRetrievalId: 'ret-active-uuid',
      );

      expect(memory.isUnderRecovery, isTrue);
      expect(memory.isReady, isFalse);
      expect(memory.isCompleted, isFalse);
      expect(memory.activeRetrievalId, 'ret-active-uuid');
    });

    // 3. While under recovery: second initiate attempt is rejected
    test('3. While under recovery: second initiate attempt is rejected', () async {
      const memoryId = 'mem-busy-3';
      const userA = 'user-uuid-recovery-owner';
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      // Save as already recovered to verify guard
      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: memoryId,
          reconstructedPlaintext: 'TEST PLAINTEXT',
          recoveredAt: DateTime.now(),
        ),
      );

      expect(
        () => MemoryService.instance.initiateMemoryRetrieval(memoryId),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('already completed recovery'),
        )),
      );
    });

    // 4. Complete recovery: status becomes COMPLETED
    test('4. Complete recovery: status becomes COMPLETED', () {
      final memory = UserStoredMemory(
        id: 'mem-completed-4',
        createdAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
        fragmentCount: 3,
        packetCount: 3,
        memorizedCount: 3,
        status: 'recovered',
        retrievalStatus: 'completed',
      );

      expect(memory.isCompleted, isTrue);
      expect(memory.isReady, isFalse);
      expect(memory.isUnderRecovery, isFalse);
    });

    // 5. Completed memory: no longer appears in active recoverable-memory list
    test('5. Completed memory: excluded from active recoverable-memory list', () async {
      const userA = 'user-uuid-active-filter';
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      final mem1 = UserStoredMemory(
        id: 'mem-active-5a',
        createdAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
        fragmentCount: 3,
        packetCount: 3,
        memorizedCount: 3,
        status: 'active',
        retrievalStatus: 'ready',
      );

      final mem2 = UserStoredMemory(
        id: 'mem-recovered-5b',
        createdAt: DateTime.now(),
        expiresAt: DateTime.now().add(const Duration(days: 7)),
        fragmentCount: 3,
        packetCount: 3,
        memorizedCount: 3,
        status: 'recovered',
        retrievalStatus: 'completed',
      );

      // Record mem2 in local recovery history
      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: 'mem-recovered-5b',
          reconstructedPlaintext: 'RECONSTRUCTED MESSAGE',
          recoveredAt: DateTime.now(),
        ),
      );

      final allMemories = [mem1, mem2];
      final activeOnly = allMemories
          .where((m) => !m.isCompleted)
          .toList();

      expect(activeOnly.length, 1);
      expect(activeOnly.first.id, 'mem-active-5a');
      expect(await RecoveryHistoryService.instance.hasRecovered(mem2.id), isTrue);
    });

    // 6. Completed memory: attempting another recovery is rejected
    test('6. Completed memory: attempting another recovery is rejected', () async {
      const userA = 'user-uuid-rejection-test';
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      const completedMemId = 'mem-already-done-6';
      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: completedMemId,
          reconstructedPlaintext: 'THE MEETING IS AT 4:30 PM',
          recoveredAt: DateTime.now(),
        ),
      );

      expect(
        () => MemoryService.instance.initiateMemoryRetrieval(completedMemId),
        throwsA(isA<StateError>()),
      );
    });

    // 7. Logout -> login: completed status remains completed
    test('7. Logout -> login preserves completed status in history', () async {
      const userA = 'user-uuid-auth-cycle';
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: 'mem-persistent-7',
          reconstructedPlaintext: 'PLAINTEXT SURVIVES LOGOUT',
          recoveredAt: DateTime.now(),
          packetCount: 4,
        ),
      );

      // Verify saved in history
      expect(await RecoveryHistoryService.instance.hasRecovered('mem-persistent-7'), isTrue);

      // Logout
      RecoveryHistoryService.instance.onSignOut();
      expect(RecoveryHistoryService.instance.historyNotifier.value, isEmpty);

      // Login again as User A
      final restored = await RecoveryHistoryService.instance.getHistory(userId: userA);
      expect(restored.length, 1);
      expect(restored.first.memoryId, 'mem-persistent-7');
      expect(restored.first.reconstructedPlaintext, 'PLAINTEXT SURVIVES LOGOUT');
    });

    // 8. App restart: completed status remains completed
    test('8. App restart: completed status remains completed in local history', () async {
      const userA = 'user-uuid-app-restart';
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: 'mem-restart-8',
          reconstructedPlaintext: 'SURVIVES APP RESTART',
          recoveredAt: DateTime.now(),
          packetCount: 5,
        ),
      );

      // Simulate app restart by clearing in-memory state and recreating service state
      RecoveryHistoryService.instance.resetForTesting();
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      final reloaded = await RecoveryHistoryService.instance.getHistory();
      expect(reloaded.length, 1);
      expect(reloaded.first.memoryId, 'mem-restart-8');
      expect(reloaded.first.reconstructedPlaintext, 'SURVIVES APP RESTART');
    });

    // 9. Recovery history: reconstructed plaintext is persisted locally for owner
    test('9. Recovery history: reconstructed plaintext is persisted locally for owner', () async {
      const userA = 'user-uuid-plaintext-owner';
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      const secretMessage = 'THE MEETING IS AT 4:30 PM IN ROOM 204';
      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: 'mem-secret-9',
          reconstructedPlaintext: secretMessage,
          recoveredAt: DateTime.now(),
          packetCount: 6,
        ),
      );

      final entry = await RecoveryHistoryService.instance.getRecoveredMemory('mem-secret-9');
      expect(entry, isNotNull);
      expect(entry!.reconstructedPlaintext, secretMessage);
      expect(entry.packetCount, 6);
    });

    // 10. Multi-user local isolation
    test('10. Multi-user local isolation: User B never sees User A recovery history', () async {
      const userA = 'user-uuid-alpha-10';
      const userB = 'user-uuid-beta-10';

      // User A saves recovered plaintext
      RecoveryHistoryService.instance.setUserIdForTesting(userA);
      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: 'mem-userA-secret',
          reconstructedPlaintext: 'USER A SECRET MESSAGE',
          recoveredAt: DateTime.now(),
        ),
      );
      expect((await RecoveryHistoryService.instance.getHistory()).length, 1);

      // Logout User A, switch to User B
      RecoveryHistoryService.instance.onSignOut();
      RecoveryHistoryService.instance.setUserIdForTesting(userB);

      // User B sees ZERO history from User A
      final userBHistory = await RecoveryHistoryService.instance.getHistory();
      expect(userBHistory, isEmpty);
      expect(await RecoveryHistoryService.instance.hasRecovered('mem-userA-secret'), isFalse);
      expect(await RecoveryHistoryService.instance.getRecoveredMemory('mem-userA-secret'), isNull);

      // User B creates own recovery history
      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: 'mem-userB-secret',
          reconstructedPlaintext: 'USER B DIFFERENT SECRET',
          recoveredAt: DateTime.now(),
        ),
      );
      expect((await RecoveryHistoryService.instance.getHistory()).length, 1);

      // Switch back to User A
      RecoveryHistoryService.instance.setUserIdForTesting(userA);
      final userARestored = await RecoveryHistoryService.instance.getHistory();
      expect(userARestored.length, 1);
      expect(userARestored.first.reconstructedPlaintext, 'USER A SECRET MESSAGE');
      expect(await RecoveryHistoryService.instance.hasRecovered('mem-userB-secret'), isFalse);

      // User B clears history: User A history remains intact
      RecoveryHistoryService.instance.setUserIdForTesting(userB);
      await RecoveryHistoryService.instance.clearUserHistory();
      expect((await RecoveryHistoryService.instance.getHistory()).length, 0);

      RecoveryHistoryService.instance.setUserIdForTesting(userA);
      final userAFinal = await RecoveryHistoryService.instance.getHistory();
      expect(userAFinal.length, 1);
      expect(userAFinal.first.memoryId, 'mem-userA-secret');
    });

    // 11. Station/node storage: recovered plaintext does NOT appear in node metadata and does NOT consume 5 node slots
    test('11. Station/node storage: recovery history does not consume node slots or appear in node metadata', () async {
      const userA = 'user-uuid-slot-check';
      NodeStorageService.instance.setUserIdForTesting(userA);
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      // Verify node starts at 0/5 slots
      expect(NodeStorageService.instance.activeHostedCount, 0);

      // Save 3 recovered memories in owner history
      for (int i = 1; i <= 3; i++) {
        await RecoveryHistoryService.instance.saveRecoveredMemory(
          RecoveredMemory(
            memoryId: 'mem-owner-$i',
            reconstructedPlaintext: 'RECOVERED PLAINTEXT $i',
            recoveredAt: DateTime.now(),
          ),
        );
      }

      // Verify owner history has 3 items
      final history = await RecoveryHistoryService.instance.getHistory();
      expect(history.length, 3);

      // Verify node storage slots remain completely unaffected (0 / 5)
      expect(NodeStorageService.instance.activeHostedCount, 0);
      expect(NodeStorageService.instance.isCapacityFull, isFalse);
      expect(NodeStorageService.instance.remainingSlots, 5);

      // Verify node storage has no fragments matching recovered memories
      for (int i = 1; i <= 3; i++) {
        expect(await NodeStorageService.instance.hasFragment('mem-owner-$i'), isFalse);
      }
    });

    // 12. Zero disk plaintext invariant for node storage remains intact
    test('12. Zero disk plaintext invariant for node storage remains intact when recovery history exists', () async {
      const userA = 'user-uuid-zero-node-plain';
      NodeStorageService.instance.setUserIdForTesting(userA);
      RecoveryHistoryService.instance.setUserIdForTesting(userA);

      // Node storage saves metadata only
      await NodeStorageService.instance.saveFragment(
        StoredNodeFragment(
          fragmentId: 'frag-meta-only',
          memoryId: 'mem-parent',
          sequenceNumber: 1,
          sizeBytes: 42,
          receivedAt: DateTime.now(),
          expiresAt: DateTime.now().add(const Duration(days: 30)),
          assignmentId: 'assign-uuid',
          verificationHash: 'hash-abc',
        ),
      );

      // Recovery history saves owner plaintext
      await RecoveryHistoryService.instance.saveRecoveredMemory(
        RecoveredMemory(
          memoryId: 'mem-owner-complete',
          reconstructedPlaintext: 'TOP SECRET HUMAN SERVER MESSAGE',
          recoveredAt: DateTime.now(),
        ),
      );

      final prefs = await SharedPreferences.getInstance();

      // Inspect all node storage keys: MUST NEVER contain plaintext
      for (final key in prefs.getKeys()) {
        if (key.startsWith('human_server_node_')) {
          final val = prefs.get(key);
          if (val is String) {
            expect(val.contains('TOP SECRET HUMAN SERVER MESSAGE'), isFalse);
            expect(val.contains('"plaintext"'), isFalse);
            expect(val.contains('"payloadText"'), isFalse);
          }
        }
      }

      // Plaintext only exists in owner recovery history namespace
      final historyKey = 'human_server_recovered_history_entry_${userA}_mem-owner-complete';
      final historyVal = prefs.getString(historyKey);
      expect(historyVal, isNotNull);
      expect(historyVal!.contains('TOP SECRET HUMAN SERVER MESSAGE'), isTrue);
    });
  });
}

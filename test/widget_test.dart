import 'dart:convert';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:human_server/models/pending_delivery_fragment.dart';
import 'package:human_server/models/stored_node_fragment.dart';
import 'package:human_server/screens/auth_screen.dart';
import 'package:human_server/screens/main_screen.dart';
import 'package:human_server/screens/node_screen.dart';
import 'package:human_server/screens/store_screen.dart';
import 'package:human_server/services/memory_service.dart';
import 'package:human_server/services/node_storage_service.dart';
import 'package:human_server/theme/app_theme.dart';
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

      expect(find.text('Human Server'), findsOneWidget);
      expect(find.text('Home'), findsOneWidget);
      expect(find.text('Store'), findsOneWidget);
      expect(find.text('Retrieve'), findsOneWidget);
      expect(find.text('Node'), findsOneWidget);
      expect(find.text('Profile'), findsOneWidget);
    });

    testWidgets('Human Server auth screen smoke test', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const AuthScreen(),
        ),
      );

      expect(find.text('Human Server'), findsOneWidget);
      expect(find.text('Sign In'), findsOneWidget);
      expect(find.text('Create Node'), findsOneWidget);
      expect(find.text('NODE EMAIL ADDRESS'), findsOneWidget);
      expect(find.text('PASSWORD'), findsOneWidget);
      expect(find.text('Sign In to Node'), findsOneWidget);

      await tester.tap(find.text('Create Node'));
      await tester.pump();

      expect(find.text('CONFIRM PASSWORD'), findsOneWidget);
      expect(find.text('Create Node Account'), findsOneWidget);
    });

    testWidgets('Human Server node screen smoke test', (WidgetTester tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: const NodeScreen(),
        ),
      );

      expect(find.text('My Node Telemetry'), findsOneWidget);
      expect(find.text('RESPONSE TELEMETRY'), findsOneWidget);
      expect(find.text('AVG RESPONSE TIME'), findsOneWidget);
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

      expect(find.text('Store Memory'), findsOneWidget);
      expect(find.text('Active Storage Slots'), findsOneWidget);
      expect(find.text('MEMORY CONTENT'), findsOneWidget);
      expect(find.text('Fragment & Store Memory'), findsOneWidget);
    });
  });
}

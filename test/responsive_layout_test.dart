import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:human_server/screens/auth_screen.dart';
import 'package:human_server/screens/home_screen.dart';
import 'package:human_server/screens/main_screen.dart';
import 'package:human_server/screens/node_screen.dart';
import 'package:human_server/screens/profile_screen.dart';
import 'package:human_server/screens/retrieve_screen.dart';
import 'package:human_server/screens/store_screen.dart';
import 'package:human_server/theme/app_theme.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  const testWidths = [320.0, 360.0, 390.0, 412.0, 480.0, 1200.0];

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    FlutterError.onError = (details) {
      FlutterError.dumpErrorToConsole(details);
    };
  });

  group('Mobile & Desktop Responsive Layout Verification', () {
    for (final width in testWidths) {
      testWidgets('HomeScreen renders cleanly without overflow at ${width}dp',
          (WidgetTester tester) async {
        tester.view.physicalSize = Size(width, 800);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);

        await tester.pumpWidget(
          MaterialApp(
            theme: AppTheme.darkTheme,
            home: Scaffold(
              body: HomeScreen(
                onNavigateToStore: () {},
                onNavigateToRetrieve: () {},
              ),
            ),
          ),
        );
        await tester.pump();

        expect(tester.takeException(), isNull,
            reason: 'HomeScreen must not throw any overflow or layout exceptions at ${width}dp');
      });

      testWidgets('MainScreen renders cleanly without overflow at ${width}dp',
          (WidgetTester tester) async {
        tester.view.physicalSize = Size(width, 800);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);

        await tester.pumpWidget(
          MaterialApp(
            theme: AppTheme.darkTheme,
            home: const MainScreen(),
          ),
        );
        await tester.pump();

        expect(tester.takeException(), isNull,
            reason: 'MainScreen must not throw any overflow or layout exceptions at ${width}dp');
      });

      testWidgets('NodeScreen renders cleanly without overflow at ${width}dp',
          (WidgetTester tester) async {
        tester.view.physicalSize = Size(width, 800);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);

        String? overflowSummary;
        final originalOnError = FlutterError.onError;
        FlutterError.onError = (details) {
          overflowSummary = details.toString();
        };

        await tester.pumpWidget(
          MaterialApp(
            theme: AppTheme.darkTheme,
            home: const NodeScreen(),
          ),
        );
        await tester.pump();
        FlutterError.onError = originalOnError;

        expect(overflowSummary, isNull,
            reason: 'NodeScreen must not throw any overflow or layout exceptions at ${width}dp');
      });

      testWidgets('StoreScreen renders cleanly without overflow at ${width}dp',
          (WidgetTester tester) async {
        tester.view.physicalSize = Size(width, 800);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);

        String? overflowSummary;
        final originalOnError = FlutterError.onError;
        FlutterError.onError = (details) {
          overflowSummary = details.toString();
        };

        await tester.pumpWidget(
          MaterialApp(
            theme: AppTheme.darkTheme,
            home: const StoreScreen(),
          ),
        );
        await tester.pump();
        FlutterError.onError = originalOnError;

        expect(overflowSummary, isNull,
            reason: 'StoreScreen must not throw any overflow or layout exceptions at ${width}dp');
      });

      testWidgets('RetrieveScreen renders cleanly without overflow at ${width}dp',
          (WidgetTester tester) async {
        tester.view.physicalSize = Size(width, 800);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);

        await tester.pumpWidget(
          MaterialApp(
            theme: AppTheme.darkTheme,
            home: const RetrieveScreen(),
          ),
        );
        await tester.pump();

        expect(tester.takeException(), isNull,
            reason: 'RetrieveScreen must not throw any overflow or layout exceptions at ${width}dp');
      });

      testWidgets('ProfileScreen renders cleanly without overflow at ${width}dp',
          (WidgetTester tester) async {
        tester.view.physicalSize = Size(width, 800);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);

        await tester.pumpWidget(
          MaterialApp(
            theme: AppTheme.darkTheme,
            home: const ProfileScreen(),
          ),
        );
        await tester.pump();

        expect(tester.takeException(), isNull,
            reason: 'ProfileScreen must not throw any overflow or layout exceptions at ${width}dp');
      });

      testWidgets('AuthScreen renders cleanly without overflow at ${width}dp',
          (WidgetTester tester) async {
        tester.view.physicalSize = Size(width, 800);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);
        addTearDown(tester.view.resetDevicePixelRatio);

        await tester.pumpWidget(
          MaterialApp(
            theme: AppTheme.darkTheme,
            home: const AuthScreen(),
          ),
        );
        await tester.pump();

        expect(tester.takeException(), isNull,
            reason: 'AuthScreen must not throw any overflow or layout exceptions at ${width}dp');
      });
    }
  });
}

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../services/auth_service.dart';
import '../services/node_service.dart';
import '../theme/app_theme.dart';
import 'auth_screen.dart';
import 'main_screen.dart';

class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  late final StreamSubscription<AuthState> _authSubscription;
  bool _isLoading = true;
  bool _hasSession = false;

  @override
  void initState() {
    super.initState();
    _checkInitialSession();

    // Listen to Supabase auth events (signed in, signed out, token refreshed)
    _authSubscription = AuthService.instance.authStateChanges.listen((data) {
      final session = data.session;
      final hasSession = session != null;

      if (mounted) {
        setState(() {
          _hasSession = hasSession;
          _isLoading = false;
        });
      }

      if (hasSession && session.user.id.isNotEmpty) {
        AuthService.instance.ensureProfileExists(session.user).then((_) {
          NodeService.instance.registerNodeIfNeeded().then((_) {
            NodeService.instance.syncNode();
          });
        });
      } else {
        NodeService.instance.clear();
      }
    });
  }

  void _checkInitialSession() {
    final session = AuthService.instance.currentSession;
    final hasSession = session != null;

    setState(() {
      _hasSession = hasSession;
      _isLoading = false;
    });

    if (hasSession && session.user.id.isNotEmpty) {
      AuthService.instance.ensureProfileExists(session.user).then((_) {
        NodeService.instance.registerNodeIfNeeded().then((_) {
          NodeService.instance.syncNode();
        });
      });
    } else {
      NodeService.instance.clear();
    }
  }

  @override
  void dispose() {
    _authSubscription.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        backgroundColor: AppTheme.background,
        body: Center(
          child: CircularProgressIndicator(
            color: AppTheme.primary,
          ),
        ),
      );
    }

    if (_hasSession) {
      return const MainScreen();
    }

    return const AuthScreen();
  }
}

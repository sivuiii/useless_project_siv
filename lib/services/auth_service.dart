import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/user_profile.dart';
import 'node_service.dart';
import 'node_storage_service.dart';

class AuthService {
  AuthService._();
  static final AuthService instance = AuthService._();

  SupabaseClient? get _supabase {
    try {
      return Supabase.instance.client;
    } catch (_) {
      return null;
    }
  }

  User? get currentUser => _supabase?.auth.currentUser;
  Session? get currentSession => _supabase?.auth.currentSession;
  bool get isAuthenticated => currentSession != null;

  Stream<AuthState> get authStateChanges =>
      _supabase?.auth.onAuthStateChange ?? const Stream<AuthState>.empty();

  /// Holds the active user profile in memory and notifies listeners when updated
  final ValueNotifier<UserProfile?> currentProfileNotifier =
      ValueNotifier<UserProfile?>(null);

  UserProfile? get currentProfile => currentProfileNotifier.value;

  /// Sign up with email and password
  Future<AuthResponse> signUp({
    required String email,
    required String password,
  }) async {
    final client = _supabase;
    if (client == null) {
      throw const AuthException('Supabase is not initialized');
    }

    final response = await client.auth.signUp(
      email: email.trim(),
      password: password,
    );

    if (response.user != null) {
      await ensureProfileExists(response.user!);
      await NodeService.instance.registerNodeIfNeeded();
      await NodeStorageService.instance.getStoredFragments(userId: response.user!.id);
    }

    return response;
  }

  /// Sign in with email and password
  Future<AuthResponse> signIn({
    required String email,
    required String password,
  }) async {
    final client = _supabase;
    if (client == null) {
      throw const AuthException('Supabase is not initialized');
    }

    final response = await client.auth.signInWithPassword(
      email: email.trim(),
      password: password,
    );

    if (response.user != null) {
      await ensureProfileExists(response.user!);
      await NodeService.instance.registerNodeIfNeeded();
      await NodeStorageService.instance.getStoredFragments(userId: response.user!.id);
    }

    return response;
  }

  /// Sign out the current session
  Future<void> signOut() async {
    final client = _supabase;
    if (client != null) {
      await client.auth.signOut();
    }
    currentProfileNotifier.value = null;
    NodeService.instance.clear();
    await NodeStorageService.instance.onSignOut();
  }

  /// Ensures a profile row exists in the `profiles` table.
  /// If it already exists, fetches it without modifying existing values.
  /// If it doesn't exist, inserts a new profile row.
  Future<UserProfile?> ensureProfileExists(User user) async {
    final client = _supabase;
    if (client == null) return null;

    try {
      // 1. Check if profile already exists
      final existingData = await client
          .from('profiles')
          .select()
          .eq('id', user.id)
          .maybeSingle();

      if (existingData != null) {
        final profile = UserProfile.fromMap(existingData);
        currentProfileNotifier.value = profile;
        return profile;
      }

      // 2. Derive initial username from email before the '@' symbol
      final email = user.email ?? '';
      String derivedUsername = 'node_user';
      if (email.contains('@')) {
        final prefix = email.split('@').first.trim();
        if (prefix.isNotEmpty) {
          derivedUsername = prefix;
        }
      }

      final initialProfile = {
        'id': user.id,
        'username': derivedUsername,
        'credits': 0,
        'reliability': 1.0,
      };

      // 3. Upsert new profile into Supabase (compatible with trigger & manual creation)
      final insertedData = await client
          .from('profiles')
          .upsert(initialProfile)
          .select()
          .single();

      final profile = UserProfile.fromMap(insertedData);
      currentProfileNotifier.value = profile;
      return profile;
    } catch (e) {
      debugPrint('Error syncing profile for user ${user.id}: $e');
      // If table insert fails (e.g., RLS policy not set yet or network glitch),
      // provide an in-memory fallback so user can still navigate the app
      final fallbackProfile = UserProfile(
        id: user.id,
        username: user.email?.split('@').first ?? 'node_user',
        createdAt: DateTime.now(),
        credits: 0,
        reliability: 1.0,
      );
      currentProfileNotifier.value = fallbackProfile;
      return fallbackProfile;
    }
  }

  /// Refreshes the profile from Supabase
  Future<UserProfile?> refreshProfile() async {
    final user = currentUser;
    if (user == null) {
      return null;
    }
    return ensureProfileExists(user);
  }
}

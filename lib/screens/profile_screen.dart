import 'package:flutter/material.dart';

import '../models/mock_data.dart';
import '../models/node.dart';
import '../models/user_profile.dart';
import '../services/auth_service.dart';
import '../services/node_service.dart';
import '../theme/app_theme.dart';
import '../widgets/max_width_container.dart';
import '../widgets/status_badge.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  bool _isLoggingOut = false;

  @override
  void initState() {
    super.initState();
    // Refresh profile and node in background to get latest data from Supabase
    AuthService.instance.refreshProfile();
    NodeService.instance.getCurrentNode();
  }

  void _showLogoutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: AppTheme.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text(
          'Disconnect Node',
          style: TextStyle(
            color: AppTheme.textPrimary,
            fontWeight: FontWeight.w700,
          ),
        ),
        content: const Text(
          'Are you sure you want to sign out? Your node will stop serving memory challenges until you reconnect.',
          style: TextStyle(color: AppTheme.textSecondary, fontSize: 13),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel', style: TextStyle(color: AppTheme.textMuted)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.error),
            onPressed: () async {
              final messenger = ScaffoldMessenger.of(context);
              Navigator.pop(dialogContext);
              setState(() {
                _isLoggingOut = true;
              });
              try {
                await AuthService.instance.signOut();
              } catch (e) {
                if (mounted) {
                  messenger.showSnackBar(
                    SnackBar(
                      content: Text('Error signing out: $e'),
                      backgroundColor: AppTheme.error,
                    ),
                  );
                  setState(() {
                    _isLoggingOut = false;
                  });
                }
              }
            },
            child: const Text('Sign Out'),
          ),
        ],
      ),
    );
  }

  String _formatJoinDate(DateTime date) {
    final months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return '${months[date.month - 1]} ${date.year}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile & Settings'),
      ),
      body: SingleChildScrollView(
        child: MaxWidthContainer(
          child: ValueListenableBuilder<UserProfile?>(
            valueListenable: AuthService.instance.currentProfileNotifier,
            builder: (context, profile, _) {
              final user = MockDataRepository.currentUser;
              final username = profile?.username.isNotEmpty == true
                  ? profile!.username
                  : (AuthService.instance.currentUser?.email?.split('@').first ??
                      user.username);
              final credits = profile?.credits ?? 0;
              final reliabilityVal = profile != null
                  ? (profile.reliability <= 1.0
                      ? (profile.reliability * 100).toStringAsFixed(1)
                      : profile.reliability.toStringAsFixed(1))
                  : '${user.reliability}';
              final joinDate = profile != null
                  ? _formatJoinDate(profile.createdAt)
                  : user.joinDate;

              return ValueListenableBuilder<NodeModel?>(
                valueListenable: NodeService.instance.currentNodeNotifier,
                builder: (context, node, _) {
                  final isOnline = node?.isOnline ?? true;
                  final nodeId = node != null && node.id.isNotEmpty
                      ? (node.id.length >= 8
                          ? 'HS-NODE-${node.id.substring(0, 8).toUpperCase()}'
                          : node.id)
                      : (profile != null && profile.id.isNotEmpty
                          ? 'HS-NODE-${profile.id.substring(0, 8).toUpperCase()}'
                          : user.nodeId);

                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Profile Header Card
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          color: AppTheme.surface,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: Column(
                          children: [
                            Row(
                              children: [
                                CircleAvatar(
                                  radius: 28,
                                  backgroundColor:
                                      AppTheme.primary.withValues(alpha: 0.2),
                                  child: Text(
                                    username.isNotEmpty
                                        ? username.substring(0, username.length >= 2 ? 2 : 1).toUpperCase()
                                        : 'ND',
                                    style: const TextStyle(
                                      color: AppTheme.primaryLight,
                                      fontSize: 18,
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 16),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        username,
                                        style: const TextStyle(
                                          color: AppTheme.textPrimary,
                                          fontSize: 18,
                                          fontWeight: FontWeight.w700,
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        'Member since $joinDate • $nodeId',
                                        style: const TextStyle(
                                          color: AppTheme.textMuted,
                                          fontSize: 12,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                StatusBadge(
                                  isOnline: isOnline,
                                  label: isOnline ? 'ONLINE' : 'OFFLINE',
                                ),
                              ],
                            ),
                        const SizedBox(height: 18),
                        const Divider(),
                        const SizedBox(height: 10),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceAround,
                          children: [
                            _buildAccountMetric(
                              'CREDITS',
                              '$credits CR',
                              AppTheme.secondary,
                            ),
                            _buildAccountMetric(
                              'RELIABILITY',
                              '$reliabilityVal%',
                              AppTheme.primary,
                            ),
                            _buildAccountMetric(
                              'STORAGE LIMIT',
                              '${user.maxMemoriesLimit} Items',
                              AppTheme.warning,
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),

                  // Account Section
                  const Text(
                    'ACCOUNT OVERVIEW',
                    style: TextStyle(
                      color: AppTheme.textMuted,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.8,
                    ),
                  ),
                  const SizedBox(height: 10),

                  _buildSettingsTile(
                    icon: Icons.person_outline_rounded,
                    title: 'Account & Identity',
                    subtitle: AuthService.instance.currentUser?.email ??
                        'Manage node public key and profile metadata',
                    onTap: () {},
                  ),
                  _buildSettingsTile(
                    icon: Icons.account_balance_wallet_outlined,
                    title: 'Credits & Earnings',
                    subtitle: 'Current balance: $credits Credits',
                    onTap: () {},
                  ),

                  const SizedBox(height: 24),

                  // Settings Placeholders Section
                  const Text(
                    'NODE SETTINGS',
                    style: TextStyle(
                      color: AppTheme.textMuted,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.8,
                    ),
                  ),
                  const SizedBox(height: 10),

                  _buildSettingsTile(
                    icon: Icons.sd_storage_outlined,
                    title: 'Node Storage Quota',
                    subtitle: 'Allocated space: 50 MB / 500 MB',
                    onTap: () {},
                  ),
                  _buildSettingsTile(
                    icon: Icons.wifi_protected_setup_rounded,
                    title: 'Network Sync Protocol',
                    subtitle: 'Sync only on Wi-Fi (Enabled)',
                    onTap: () {},
                  ),
                  _buildSettingsTile(
                    icon: Icons.security_rounded,
                    title: 'Security & Encryption',
                    subtitle: 'AES-256 local fragment encryption key',
                    onTap: () {},
                  ),
                  _buildSettingsTile(
                    icon: Icons.notifications_none_rounded,
                    title: 'Verification Challenge Alerts',
                    subtitle: 'Receive background recall ping requests',
                    onTap: () {},
                  ),

                  const SizedBox(height: 32),

                  // UI Logout Button (Real Functional Supabase Sign Out)
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: OutlinedButton.icon(
                      onPressed: _isLoggingOut ? null : () => _showLogoutDialog(context),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppTheme.error,
                        side: BorderSide(
                          color: AppTheme.error.withValues(alpha: 0.4),
                        ),
                      ),
                      icon: _isLoggingOut
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: AppTheme.error,
                              ),
                            )
                          : const Icon(Icons.logout_rounded, size: 18),
                      label: Text(
                        _isLoggingOut ? 'Disconnecting...' : 'Logout',
                        style: const TextStyle(fontWeight: FontWeight.w700),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Center(
                    child: Text(
                      'Human Server v1.0.0 (MVP)',
                      style: TextStyle(color: AppTheme.textMuted, fontSize: 11),
                    ),
                  ),
                ],
              );
            },
          );
        },
      ),
    ),
  ),
);
}

  Widget _buildAccountMetric(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(
            color: AppTheme.textMuted,
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.8,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 15,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }

  Widget _buildSettingsTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.border),
      ),
      child: ListTile(
        onTap: onTap,
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: AppTheme.surfaceLight,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: AppTheme.textPrimary, size: 20),
        ),
        title: Text(
          title,
          style: const TextStyle(
            color: AppTheme.textPrimary,
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
        subtitle: Text(
          subtitle,
          style: const TextStyle(
            color: AppTheme.textMuted,
            fontSize: 12,
          ),
        ),
        trailing: const Icon(Icons.chevron_right_rounded,
            color: AppTheme.textMuted, size: 20),
      ),
    );
  }
}

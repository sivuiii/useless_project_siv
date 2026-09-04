import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

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
    AuthService.instance.refreshProfile();
    NodeService.instance.getCurrentNode();
  }

  void _showLogoutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: AppTheme.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
          side: const BorderSide(color: AppTheme.border, width: 1.0),
        ),
        title: Text(
          'DISCONNECT OPERATOR STATION',
          style: GoogleFonts.playfairDisplay(
            color: AppTheme.textPrimary,
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
        content: Text(
          'Disconnecting will pause your station. You will stop receiving biological memory fragment challenges until re-authenticated.',
          style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 13),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: Text('CANCEL', style: GoogleFonts.spaceMono(color: AppTheme.textMuted, fontSize: 11)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.signalOffline,
              side: const BorderSide(color: AppTheme.signalOffline),
            ),
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
                      content: Text('DISCONNECT ERROR: $e', style: GoogleFonts.spaceMono(fontSize: 11)),
                      backgroundColor: AppTheme.signalOffline,
                    ),
                  );
                  setState(() {
                    _isLoggingOut = false;
                  });
                }
              }
            },
            child: Text('DISCONNECT', style: GoogleFonts.spaceMono(fontSize: 11, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
  }

  String _formatJoinDate(DateTime date) {
    final months = [
      'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
      'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
    ];
    return '${months[date.month - 1]} ${date.year}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'OPERATOR REGISTER & CREDENTIALS',
          style: GoogleFonts.playfairDisplay(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.2,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
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
                          ? 'HS-STATION-${node.id.substring(0, 8).toUpperCase()}'
                          : node.id)
                      : (profile != null && profile.id.isNotEmpty
                          ? 'HS-STATION-${profile.id.substring(0, 8).toUpperCase()}'
                          : user.nodeId);

                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Official Credentials Passport Card
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: AppTheme.surface,
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: AppTheme.border, width: 1.0),
                        ),
                        child: Column(
                          children: [
                            Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Container(
                                  width: 48,
                                  height: 48,
                                  decoration: BoxDecoration(
                                    color: AppTheme.surfaceElevated,
                                    border: Border.all(color: AppTheme.borderSubtle, width: 1.0),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: Center(
                                    child: Text(
                                      username.isNotEmpty
                                          ? username.substring(0, username.length >= 2 ? 2 : 1).toUpperCase()
                                          : 'ND',
                                      style: GoogleFonts.playfairDisplay(
                                        color: AppTheme.brassAccent,
                                        fontSize: 20,
                                        fontWeight: FontWeight.w800,
                                      ),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 14),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        username.toUpperCase(),
                                        style: GoogleFonts.playfairDisplay(
                                          color: AppTheme.textPrimary,
                                          fontSize: 17,
                                          fontWeight: FontWeight.w800,
                                        ),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                      const SizedBox(height: 3),
                                      Text(
                                        'STATION OPERATOR • $nodeId',
                                        style: GoogleFonts.spaceMono(
                                          color: AppTheme.textMuted,
                                          fontSize: 10,
                                          letterSpacing: 0.3,
                                        ),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                      const SizedBox(height: 2),
                                      Text(
                                        'ENLISTED $joinDate',
                                        style: GoogleFonts.inter(
                                          color: AppTheme.textMuted,
                                          fontSize: 10,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                const SizedBox(width: 8),
                                StatusBadge(
                                  isOnline: isOnline,
                                  label: isOnline ? 'ONLINE' : 'OFFLINE',
                                ),
                              ],
                            ),
                            const SizedBox(height: 14),
                            const Divider(height: 1),
                            const SizedBox(height: 12),
                            Row(
                              children: [
                                Expanded(child: _buildAccountMetric('CREDITS', '$credits CR', AppTheme.brassAccent)),
                                Container(width: 1, height: 28, color: AppTheme.border),
                                Expanded(child: _buildAccountMetric('RETENTION', '$reliabilityVal%', AppTheme.signalOnline)),
                                Container(width: 1, height: 28, color: AppTheme.border),
                                Expanded(child: _buildAccountMetric('CAPACITY', '${user.maxMemoriesLimit} ITEMS', AppTheme.signalWarning)),
                              ],
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 24),

                      Text(
                        'INSTITUTIONAL ACCOUNT REGISTER',
                        style: GoogleFonts.inter(
                          color: AppTheme.textMuted,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                        ),
                      ),
                      const SizedBox(height: 10),

                      _buildSettingsTile(
                        icon: Icons.badge_outlined,
                        title: 'OPERATOR CREDENTIALS',
                        subtitle: AuthService.instance.currentUser?.email ?? 'Circuit public key & operator identity metadata',
                        onTap: () {},
                      ),
                      _buildSettingsTile(
                        icon: Icons.account_balance_wallet_outlined,
                        title: 'CREDIT REGISTER & EARNINGS',
                        subtitle: 'Current institutional balance: $credits Credits',
                        onTap: () {},
                      ),

                      const SizedBox(height: 24),

                      Text(
                        'STATION CONFIGURATION',
                        style: GoogleFonts.inter(
                          color: AppTheme.textMuted,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                        ),
                      ),
                      const SizedBox(height: 10),

                      _buildSettingsTile(
                        icon: Icons.sd_storage_outlined,
                        title: 'STORAGE ALLOCATION QUOTA',
                        subtitle: 'Allocated biological capacity: 50 MB / 500 MB',
                        onTap: () {},
                      ),
                      _buildSettingsTile(
                        icon: Icons.wifi_protected_setup_rounded,
                        title: 'NETWORK SYNC PROTOCOL',
                        subtitle: 'Sync only on Wi-Fi (Enabled)',
                        onTap: () {},
                      ),
                      _buildSettingsTile(
                        icon: Icons.security_rounded,
                        title: 'ENCRYPTION & KEYS',
                        subtitle: 'AES-256 local fragment encryption key',
                        onTap: () {},
                      ),
                      _buildSettingsTile(
                        icon: Icons.notifications_none_rounded,
                        title: 'CHALLENGE ALERTS',
                        subtitle: 'Receive background recall ping requests',
                        onTap: () {},
                      ),

                      const SizedBox(height: 28),

                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: OutlinedButton.icon(
                          onPressed: _isLoggingOut ? null : () => _showLogoutDialog(context),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: AppTheme.signalOffline,
                            side: const BorderSide(color: AppTheme.signalOffline, width: 1.0),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                          ),
                          icon: _isLoggingOut
                              ? const SizedBox(
                                  width: 14,
                                  height: 14,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: AppTheme.signalOffline,
                                  ),
                                )
                              : const Icon(Icons.logout_rounded, size: 16),
                          label: Text(
                            _isLoggingOut ? 'DISCONNECTING...' : 'DISCONNECT OPERATOR STATION',
                            style: GoogleFonts.inter(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.6,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Center(
                        child: Text(
                          'HUMAN SERVER ARCHIVAL TERMINAL v1.0.0',
                          style: GoogleFonts.spaceMono(color: AppTheme.textMuted, fontSize: 10),
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
          style: GoogleFonts.inter(
            color: AppTheme.textMuted,
            fontSize: 10,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.playfairDisplay(
            color: color,
            fontSize: 16,
            fontWeight: FontWeight.w800,
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
        borderRadius: BorderRadius.circular(3),
        border: Border.all(color: AppTheme.border),
      ),
      child: ListTile(
        onTap: onTap,
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 2),
        leading: Container(
          padding: const EdgeInsets.all(7),
          decoration: BoxDecoration(
            color: AppTheme.surfaceElevated,
            borderRadius: BorderRadius.circular(3),
            border: Border.all(color: AppTheme.border),
          ),
          child: Icon(icon, color: AppTheme.textPrimary, size: 18),
        ),
        title: Text(
          title,
          style: GoogleFonts.inter(
            color: AppTheme.textPrimary,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
        subtitle: Text(
          subtitle,
          style: GoogleFonts.inter(
            color: AppTheme.textMuted,
            fontSize: 11,
          ),
        ),
        trailing: const Icon(Icons.chevron_right_rounded, color: AppTheme.textMuted, size: 18),
      ),
    );
  }
}

import 'package:flutter/material.dart';

import '../models/mock_data.dart';
import '../models/node.dart';
import '../models/user_profile.dart';
import '../services/auth_service.dart';
import '../services/memory_service.dart';
import '../services/node_service.dart';
import '../theme/app_theme.dart';
import '../widgets/activity_tile.dart';
import '../widgets/max_width_container.dart';
import '../widgets/stat_card.dart';
import '../widgets/status_badge.dart';

class HomeScreen extends StatefulWidget {
  final VoidCallback onNavigateToStore;
  final VoidCallback onNavigateToRetrieve;

  const HomeScreen({
    super.key,
    required this.onNavigateToStore,
    required this.onNavigateToRetrieve,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    MemoryService.instance.getActiveMemoriesCount();
    NodeService.instance.getCurrentNode();
    AuthService.instance.refreshProfile();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: AppTheme.primary.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(
                Icons.hub_rounded,
                color: AppTheme.primary,
                size: 20,
              ),
            ),
            const SizedBox(width: 10),
            const Text('Human Server'),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16.0),
            child: ValueListenableBuilder<NodeModel?>(
              valueListenable: NodeService.instance.currentNodeNotifier,
              builder: (context, node, _) {
                final isOnline = node?.isOnline ?? true;
                return StatusBadge(isOnline: isOnline);
              },
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: MaxWidthContainer(
          child: ValueListenableBuilder<UserProfile?>(
            valueListenable: AuthService.instance.currentProfileNotifier,
            builder: (context, profile, _) {
              const mockUser = MockDataRepository.currentUser;
              final username = profile?.username.isNotEmpty == true
                  ? profile!.username
                  : (AuthService.instance.currentUser?.email?.split('@').first ??
                      mockUser.username);
              final credits = profile?.credits ?? 0;

              return ValueListenableBuilder<NodeModel?>(
                valueListenable: NodeService.instance.currentNodeNotifier,
                builder: (context, node, _) {
                  final reliabilityVal = node != null
                      ? (node.reliability <= 1.0
                          ? (node.reliability * 100).toStringAsFixed(1)
                          : node.reliability.toStringAsFixed(1))
                      : (profile != null
                          ? (profile.reliability <= 1.0
                              ? (profile.reliability * 100).toStringAsFixed(1)
                              : profile.reliability.toStringAsFixed(1))
                          : '${mockUser.reliability}');

                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Hero Info Banner
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(18),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              AppTheme.surface,
                              AppTheme.surfaceLight.withValues(alpha: 0.6),
                            ],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'WELCOME, ${username.toUpperCase()}',
                                  style: const TextStyle(
                                    color: AppTheme.textMuted,
                                    fontSize: 11,
                                    fontWeight: FontWeight.w700,
                                    letterSpacing: 1.0,
                                  ),
                                ),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 3),
                                  decoration: BoxDecoration(
                                    color: AppTheme.primary.withValues(alpha: 0.1),
                                    borderRadius: BorderRadius.circular(6),
                                  ),
                                  child: const Text(
                                    'DECENTRALIZED NODE',
                                    style: TextStyle(
                                      color: AppTheme.primaryLight,
                                      fontSize: 10,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 10),
                            const Text(
                              'Distributed Human Memory Network',
                              style: TextStyle(
                                color: AppTheme.textPrimary,
                                fontSize: 18,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 4),
                            const Text(
                              'Store plaintext information securely across active human nodes. Earn credits by serving as a verification node.',
                              style: TextStyle(
                                color: AppTheme.textSecondary,
                                fontSize: 13,
                                height: 1.4,
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 20),

                      // Metrics Row
                      Row(
                        children: [
                          Expanded(
                            child: StatCard(
                              label: 'Balance',
                              value: '$credits CR',
                              icon: Icons.account_balance_wallet_rounded,
                              accentColor: AppTheme.secondary,
                              subtitle: profile != null
                                  ? 'Active node balance'
                                  : '+20 earned today',
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: StatCard(
                              label: 'Reliability',
                              value: '$reliabilityVal%',
                              icon: Icons.verified_rounded,
                              accentColor: AppTheme.primary,
                              subtitle: 'Optimal score',
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: ValueListenableBuilder<int>(
                              valueListenable:
                                  MemoryService.instance.activeMemoriesCountNotifier,
                              builder: (context, activeCount, _) {
                                return StatCard(
                                  label: 'Memories',
                                  value: '$activeCount/5',
                                  icon: Icons.inventory_2_rounded,
                                  accentColor: AppTheme.warning,
                                  subtitle: 'Expires 6 mos',
                                );
                              },
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 24),

                      // Prominent Action Buttons
                      const Text(
                        'QUICK ACTIONS',
                        style: TextStyle(
                          color: AppTheme.textMuted,
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                        ),
                      ),
                      const SizedBox(height: 12),

                      Row(
                        children: [
                          Expanded(
                            child: _ActionCard(
                              title: 'Store Memory',
                              description: 'Encrypt & chunk info into network',
                              icon: Icons.add_to_photos_rounded,
                              color: AppTheme.primary,
                              onTap: widget.onNavigateToStore,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: _ActionCard(
                              title: 'Retrieve Memory',
                              description: 'Query recall prompt across nodes',
                              icon: Icons.search_rounded,
                              color: AppTheme.secondary,
                              onTap: widget.onNavigateToRetrieve,
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 28),

                      // Recent Activity Section
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'RECENT ACTIVITY',
                            style: TextStyle(
                              color: AppTheme.textMuted,
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.8,
                            ),
                          ),
                          TextButton(
                            onPressed: () {},
                            child: const Text(
                              'View Log',
                              style: TextStyle(
                                color: AppTheme.primaryLight,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),

                      ...MockDataRepository.recentActivities
                          .map((activity) => ActivityTile(activity: activity)),
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
}

class _ActionCard extends StatelessWidget {
  final String title;
  final String description;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _ActionCard({
    required this.title,
    required this.description,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppTheme.surface,
      borderRadius: BorderRadius.circular(14),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: color.withValues(alpha: 0.3), width: 1.5),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: color, size: 22),
              ),
              const SizedBox(height: 12),
              Text(
                title,
                style: const TextStyle(
                  color: AppTheme.textPrimary,
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                description,
                style: const TextStyle(
                  color: AppTheme.textSecondary,
                  fontSize: 11,
                  height: 1.3,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

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
import '../widgets/telegraph_network_map.dart';

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
        centerTitle: false,
        titleSpacing: 16,
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 24,
              height: 24,
              decoration: BoxDecoration(
                color: AppTheme.surfaceElevated,
                border: Border.all(color: AppTheme.border, width: 1),
                borderRadius: BorderRadius.circular(3),
              ),
              child: const Icon(
                Icons.hub_rounded,
                color: AppTheme.brass,
                size: 14,
              ),
            ),
            const SizedBox(width: 10),
            Flexible(
              child: Text(
                'Human Server',
                style: GoogleFonts.playfairDisplay(
                  color: AppTheme.textPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.3,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
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
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
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
                      // Header: Typographic hierarchy (whitespace rather than loud box)
                      Padding(
                        padding: const EdgeInsets.only(top: 4.0, bottom: 12.0),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'COMMUNICATIONS DESK',
                                    style: GoogleFonts.inter(
                                      color: AppTheme.textMuted,
                                      fontSize: 10,
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: 0.8,
                                    ),
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    username,
                                    style: GoogleFonts.playfairDisplay(
                                      color: AppTheme.textPrimary,
                                      fontSize: 22,
                                      fontWeight: FontWeight.w700,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ],
                              ),
                            ),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                              decoration: BoxDecoration(
                                color: AppTheme.surfaceElevated,
                                border: Border.all(color: AppTheme.border),
                                borderRadius: BorderRadius.circular(2),
                              ),
                              child: Text(
                                'CIRCUIT ACTIVE',
                                style: GoogleFonts.spaceMono(
                                  color: AppTheme.signalOnline,
                                  fontSize: 9.5,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),

                      // 1. Network Topology (Hero Visualization)
                      const TelegraphNetworkMap(
                        activeNodesCount: 6,
                        networkHealth: 96.4,
                      ),

                      const SizedBox(height: 14),

                      // 2. Primary Communications Actions (Desk Controls)
                      LayoutBuilder(
                        builder: (context, constraints) {
                          final isNarrow = constraints.maxWidth < 460;
                          return isNarrow
                              ? Column(
                                  children: [
                                    _DeskActionButton(
                                      label: 'DISPATCH MEMORY',
                                      hint: 'Fragment & send telegram into human network',
                                      icon: Icons.send_rounded,
                                      accentColor: AppTheme.brass,
                                      onTap: widget.onNavigateToStore,
                                    ),
                                    const SizedBox(height: 8),
                                    _DeskActionButton(
                                      label: 'RECALL MEMORY',
                                      hint: 'Query human nodes to reconstruct payload',
                                      icon: Icons.downloading_rounded,
                                      accentColor: AppTheme.signalOnline,
                                      onTap: widget.onNavigateToRetrieve,
                                    ),
                                  ],
                                )
                              : Row(
                                  children: [
                                    Expanded(
                                      child: _DeskActionButton(
                                        label: 'DISPATCH MEMORY',
                                        hint: 'Fragment into human network',
                                        icon: Icons.send_rounded,
                                        accentColor: AppTheme.brass,
                                        onTap: widget.onNavigateToStore,
                                      ),
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: _DeskActionButton(
                                        label: 'RECALL MEMORY',
                                        hint: 'Query nodes to reconstruct',
                                        icon: Icons.downloading_rounded,
                                        accentColor: AppTheme.signalOnline,
                                        onTap: widget.onNavigateToRetrieve,
                                      ),
                                    ),
                                  ],
                                );
                        },
                      ),

                      const SizedBox(height: 18),

                      // 3. Storage Health & Ledger (Responsive Mobile vs Desktop)
                      Text(
                        'STORAGE STATE & LEDGER',
                        style: GoogleFonts.inter(
                          color: AppTheme.textMuted,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.6,
                        ),
                      ),
                      const SizedBox(height: 8),

                      LayoutBuilder(
                        builder: (context, constraints) {
                          final isMobile = constraints.maxWidth < 540;

                          if (isMobile) {
                            // Mobile Composition: 2 cards on top, 1 wide ledger card below
                            // Each card has at least 150dp width and NEVER overflows!
                            return Column(
                              children: [
                                Row(
                                  children: [
                                    Expanded(
                                      child: StatCard(
                                        label: 'Retention',
                                        value: '$reliabilityVal%',
                                        icon: Icons.verified_rounded,
                                        accentColor: AppTheme.signalOnline,
                                        subtitle: 'Optimal accuracy',
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Expanded(
                                      child: ValueListenableBuilder<int>(
                                        valueListenable: MemoryService.instance.activeMemoriesCountNotifier,
                                        builder: (context, activeCount, _) {
                                          return StatCard(
                                            label: 'Stored Memories',
                                            value: '$activeCount/5',
                                            icon: Icons.inventory_2_rounded,
                                            accentColor: AppTheme.signalWarning,
                                            subtitle: 'Expires 6 mos',
                                          );
                                        },
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                StatCard(
                                  label: 'Credit Ledger Balance',
                                  value: '$credits CR',
                                  icon: Icons.account_balance_wallet_rounded,
                                  accentColor: AppTheme.brass,
                                  subtitle: 'Institutional node balance',
                                ),
                              ],
                            );
                          } else {
                            // Desktop Composition: 3 balanced cards across the row
                            return Row(
                              children: [
                                Expanded(
                                  child: StatCard(
                                    label: 'Credit Ledger',
                                    value: '$credits CR',
                                    icon: Icons.account_balance_wallet_rounded,
                                    accentColor: AppTheme.brass,
                                    subtitle: 'Node balance',
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Expanded(
                                  child: StatCard(
                                    label: 'Retention',
                                    value: '$reliabilityVal%',
                                    icon: Icons.verified_rounded,
                                    accentColor: AppTheme.signalOnline,
                                    subtitle: 'Optimal score',
                                  ),
                                ),
                                const SizedBox(width: 10),
                                Expanded(
                                  child: ValueListenableBuilder<int>(
                                    valueListenable: MemoryService.instance.activeMemoriesCountNotifier,
                                    builder: (context, activeCount, _) {
                                      return StatCard(
                                        label: 'Stored Memories',
                                        value: '$activeCount/5',
                                        icon: Icons.inventory_2_rounded,
                                        accentColor: AppTheme.signalWarning,
                                        subtitle: 'Expires 6 mos',
                                      );
                                    },
                                  ),
                                ),
                              ],
                            );
                          }
                        },
                      ),

                      const SizedBox(height: 22),

                      // 4. Activity Ledger (Clean typographic lines, no box nesting!)
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              'COMMUNICATIONS REGISTER',
                              style: GoogleFonts.inter(
                                color: AppTheme.textMuted,
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                letterSpacing: 0.6,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            'LOG',
                            style: GoogleFonts.spaceMono(
                              color: AppTheme.textMuted,
                              fontSize: 9.5,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                      const Divider(thickness: 0.8),

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

/// Tactical communications desk button designed to feel like an instrument control.
class _DeskActionButton extends StatelessWidget {
  final String label;
  final String hint;
  final IconData icon;
  final Color accentColor;
  final VoidCallback onTap;

  const _DeskActionButton({
    required this.label,
    required this.hint,
    required this.icon,
    required this.accentColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppTheme.surfaceElevated,
      borderRadius: BorderRadius.circular(2),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(2),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(2),
            border: Border.all(color: AppTheme.border, width: 1.0),
          ),
          child: Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: AppTheme.surface,
                  borderRadius: BorderRadius.circular(2),
                  border: Border.all(color: AppTheme.border),
                ),
                child: Icon(icon, color: accentColor, size: 16),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: GoogleFonts.inter(
                        color: AppTheme.textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.3,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 1),
                    Text(
                      hint,
                      style: GoogleFonts.inter(
                        color: AppTheme.textSecondary,
                        fontSize: 11,
                        height: 1.2,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              const Icon(Icons.arrow_forward_ios_rounded, size: 12, color: AppTheme.textMuted),
            ],
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';

import '../models/mock_data.dart';
import '../models/node.dart';
import '../models/pending_delivery_fragment.dart';
import '../models/stored_node_fragment.dart';
import '../services/node_service.dart';
import '../services/node_storage_service.dart';
import '../theme/app_theme.dart';
import '../widgets/max_width_container.dart';
import '../widgets/memorization_dialog.dart';
import '../widgets/stat_card.dart';

class NodeScreen extends StatefulWidget {
  const NodeScreen({super.key});

  @override
  State<NodeScreen> createState() => _NodeScreenState();
}

class _NodeScreenState extends State<NodeScreen> {
  bool _isToggling = false;

  @override
  void initState() {
    super.initState();
    NodeStorageService.instance.getStoredFragments();
    NodeService.instance.syncNode().then((result) {
      if (mounted && result.pendingDeliveries.isNotEmpty) {
        _promptMemorization(result.pendingDeliveries.first);
      }
    });
  }

  void _promptMemorization(PendingDeliveryFragment delivery) {
    if (!mounted) return;
    MemorizationDialog.show(
      context,
      delivery: delivery,
      onMemorized: () {
        final updated = List<PendingDeliveryFragment>.from(
            NodeService.instance.pendingDeliveriesNotifier.value)
          ..removeWhere((d) => d.assignmentId == delivery.assignmentId);
        NodeService.instance.pendingDeliveriesNotifier.value = updated;

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Fragment memorized! Entrusted to your human memory.'),
            backgroundColor: AppTheme.secondary,
            duration: Duration(seconds: 3),
          ),
        );

        if (updated.isNotEmpty && mounted) {
          _promptMemorization(updated.first);
        }
      },
    );
  }

  Future<void> _handleManualSync() async {
    final result = await NodeService.instance.syncNode();
    if (mounted) {
      final msg = result.isSuccess
          ? (result.pendingDeliveries.isNotEmpty
              ? 'Synced: ${result.pendingDeliveries.length} fragment(s) awaiting memorization!'
              : (result.rebalancedCount > 0
                  ? 'Synced: ${result.rebalancedCount} fragment(s) assigned to this node'
                  : 'Node telemetry & storage up to date (${result.storedCount} memorized)'))
          : 'Sync error: ${result.error}';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(msg),
          duration: const Duration(seconds: 2),
          backgroundColor:
              result.isSuccess ? AppTheme.secondary : AppTheme.error,
        ),
      );

      if (result.pendingDeliveries.isNotEmpty) {
        _promptMemorization(result.pendingDeliveries.first);
      }
    }
  }

  Future<void> _toggleNodeStatus(bool currentOnline) async {
    if (_isToggling) return;
    setState(() {
      _isToggling = true;
    });

    final targetStatus = currentOnline ? 'offline' : 'online';

    try {
      await NodeService.instance.updateNodeStatus(targetStatus);
      if (mounted) {
        final isNowOnline = targetStatus == 'online';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              isNowOnline
                  ? 'Node status set to ONLINE'
                  : 'Node status set to OFFLINE',
            ),
            backgroundColor: isNowOnline ? AppTheme.secondary : AppTheme.error,
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to update node status: $e'),
            backgroundColor: AppTheme.error,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isToggling = false;
        });
      }
    }
  }

  String _formatLastSeen(DateTime lastSeen) {
    final diff = DateTime.now().difference(lastSeen);
    if (diff.inSeconds < 60) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }

  @override
  Widget build(BuildContext context) {
    const user = MockDataRepository.currentUser;

    return Scaffold(
      appBar: AppBar(
        title: const Text('My Node Telemetry'),
        actions: [
          ValueListenableBuilder<bool>(
            valueListenable: NodeService.instance.isSyncingNotifier,
            builder: (context, isSyncing, _) {
              return IconButton(
                icon: isSyncing
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: AppTheme.primary,
                        ),
                      )
                    : const Icon(Icons.refresh_rounded),
                onPressed: isSyncing ? null : _handleManualSync,
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            ValueListenableBuilder<bool>(
              valueListenable: NodeService.instance.isSyncingNotifier,
              builder: (context, isSyncing, _) {
                if (!isSyncing) return const SizedBox.shrink();
                return const LinearProgressIndicator(
                  minHeight: 2,
                  color: AppTheme.primary,
                  backgroundColor: AppTheme.surfaceLight,
                );
              },
            ),
            MaxWidthContainer(
          child: ValueListenableBuilder<NodeModel?>(
            valueListenable: NodeService.instance.currentNodeNotifier,
            builder: (context, node, _) {
              final isOnline = node != null ? node.isOnline : user.isNodeOnline;
              final nodeId = node != null && node.id.isNotEmpty
                  ? (node.id.length >= 8
                      ? 'HS-NODE-${node.id.substring(0, 8).toUpperCase()}'
                      : node.id)
                  : user.nodeId;
              final reliabilityVal = node != null
                  ? (node.reliability <= 1.0
                      ? (node.reliability * 100).toStringAsFixed(1)
                      : node.reliability.toStringAsFixed(1))
                  : '${user.reliability}';
              final responseRateVal = node != null
                  ? '${(node.responseRate * 100).toStringAsFixed(1)}%'
                  : '99.2%';
              final avgResponseMs = node?.avgResponseMs ?? 0;
              final lastSeenStr =
                  node != null ? _formatLastSeen(node.lastSeen) : 'Active';

              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Node Status Header Card
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: AppTheme.surface,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: isOnline
                            ? AppTheme.secondary.withValues(alpha: 0.4)
                            : AppTheme.error.withValues(alpha: 0.4),
                        width: 1.5,
                      ),
                    ),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(10),
                                  decoration: BoxDecoration(
                                    color: (isOnline
                                            ? AppTheme.secondary
                                            : AppTheme.error)
                                        .withValues(alpha: 0.15),
                                    shape: BoxShape.circle,
                                  ),
                                  child: Icon(
                                    isOnline
                                        ? Icons.dns_rounded
                                        : Icons.dns_outlined,
                                    color: isOnline
                                        ? AppTheme.secondary
                                        : AppTheme.error,
                                    size: 24,
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      nodeId,
                                      style: const TextStyle(
                                        color: AppTheme.textPrimary,
                                        fontSize: 15,
                                        fontWeight: FontWeight.w700,
                                        fontFamily: 'monospace',
                                      ),
                                    ),
                                    const SizedBox(height: 2),
                                    Text(
                                      isOnline
                                          ? 'Active & Responding'
                                          : 'Paused / Offline',
                                      style: TextStyle(
                                        color: isOnline
                                            ? AppTheme.secondary
                                            : AppTheme.error,
                                        fontSize: 12,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                            // Online / Offline Switch
                            _isToggling
                                ? const SizedBox(
                                    width: 24,
                                    height: 24,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: AppTheme.primary,
                                    ),
                                  )
                                : Switch(
                                    value: isOnline,
                                    onChanged: (_) =>
                                        _toggleNodeStatus(isOnline),
                                    activeTrackColor: AppTheme.secondary
                                        .withValues(alpha: 0.5),
                                    activeColor: AppTheme.secondary,
                                    inactiveThumbColor: AppTheme.textMuted,
                                    inactiveTrackColor: AppTheme.border,
                                  ),
                          ],
                        ),
                        const SizedBox(height: 14),
                        const Divider(),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceAround,
                          children: [
                            _buildHeaderMetric(
                              'STATUS',
                              isOnline ? 'ONLINE' : 'OFFLINE',
                              isOnline ? AppTheme.secondary : AppTheme.error,
                            ),
                            _buildHeaderMetric(
                              'RELIABILITY',
                              '$reliabilityVal%',
                              AppTheme.primary,
                            ),
                            _buildHeaderMetric(
                              'LAST SEEN',
                              lastSeenStr,
                              AppTheme.warning,
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),

                  // Response & Performance Stats
                  const Text(
                    'RESPONSE TELEMETRY',
                    style: TextStyle(
                      color: AppTheme.textMuted,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.8,
                    ),
                  ),
                  const SizedBox(height: 12),

                  Row(
                    children: [
                      Expanded(
                        child: StatCard(
                          label: 'Avg Response Time',
                          value: '$avgResponseMs ms',
                          icon: Icons.speed_rounded,
                          accentColor: AppTheme.primary,
                          subtitle: avgResponseMs > 0
                              ? 'Verified latency'
                              : 'Pending telemetry',
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: StatCard(
                          label: 'Response Rate',
                          value: responseRateVal,
                          icon: Icons.checklist_rtl_rounded,
                          accentColor: AppTheme.secondary,
                          subtitle: 'Challenge success rate',
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 10),

                  const Row(
                    children: [
                      Expanded(
                        child: StatCard(
                          label: 'Monthly Uptime',
                          value: '99.8%',
                          icon: Icons.timer_rounded,
                          accentColor: Colors.purpleAccent,
                          subtitle: '24d 18h online',
                        ),
                      ),
                      SizedBox(width: 10),
                      Expanded(
                        child: StatCard(
                          label: 'Node Rewards',
                          value: '+320 CR',
                          icon: Icons.savings_rounded,
                          accentColor: AppTheme.warning,
                          subtitle: 'Earned this month',
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 28),

                  // Pending Memorization Banner (if any deliveries waiting for human memorization)
                  ValueListenableBuilder<List<PendingDeliveryFragment>>(
                    valueListenable: NodeService.instance.pendingDeliveriesNotifier,
                    builder: (context, pending, _) {
                      if (pending.isEmpty) return const SizedBox.shrink();
                      return Container(
                        margin: const EdgeInsets.only(bottom: 18),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: AppTheme.primary.withValues(alpha: 0.12),
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                            color: AppTheme.primary.withValues(alpha: 0.6),
                            width: 1.5,
                          ),
                        ),
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: AppTheme.primary.withValues(alpha: 0.2),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(
                                Icons.psychology_rounded,
                                color: AppTheme.primary,
                                size: 22,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    '${pending.length} Fragment${pending.length > 1 ? 's' : ''} Awaiting Memorization',
                                    style: const TextStyle(
                                      color: AppTheme.textPrimary,
                                      fontSize: 13,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                  const SizedBox(height: 2),
                                  const Text(
                                    'Entrusted to your human memory.',
                                    style: TextStyle(
                                      color: AppTheme.textMuted,
                                      fontSize: 11,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            ElevatedButton(
                              onPressed: () => _promptMemorization(pending.first),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: AppTheme.primary,
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 14, vertical: 8),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                              ),
                              child: const Text(
                                'MEMORIZE',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),

                  // Fragments Memorized and Hosted in Human Memory
                  ValueListenableBuilder<List<StoredNodeFragment>>(
                    valueListenable: NodeStorageService.instance.storedFragmentsNotifier,
                    builder: (context, storedFragments, _) {
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text(
                                'MEMORIZED FRAGMENTS (HUMAN STORAGE)',
                                style: TextStyle(
                                  color: AppTheme.textMuted,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700,
                                  letterSpacing: 0.8,
                                ),
                              ),
                              Text(
                                '${storedFragments.length} Memorized',
                                style: const TextStyle(
                                  color: AppTheme.textSecondary,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),

                          if (storedFragments.isEmpty) ...[
                            Container(
                              width: double.infinity,
                              padding: const EdgeInsets.all(22),
                              decoration: BoxDecoration(
                                color: AppTheme.surface,
                                borderRadius: BorderRadius.circular(14),
                                border: Border.all(color: AppTheme.border),
                              ),
                              child: const Column(
                                children: [
                                  Icon(Icons.psychology_outlined,
                                      color: AppTheme.textMuted, size: 36),
                                  SizedBox(height: 12),
                                  Text(
                                    'No Fragments Memorized Yet',
                                    style: TextStyle(
                                      color: AppTheme.textPrimary,
                                      fontSize: 14,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  SizedBox(height: 6),
                                  Text(
                                    'This human node is online and ready. When participants store memories in the network, this node will receive fragment copies for you to read and memorize in your biological memory.',
                                    style: TextStyle(
                                      color: AppTheme.textMuted,
                                      fontSize: 12,
                                      height: 1.3,
                                    ),
                                    textAlign: TextAlign.center,
                                  ),
                                ],
                              ),
                            ),
                          ] else ...[
                            ...storedFragments.map((frag) => _buildRealFragmentCard(frag)),
                          ],
                        ],
                      );
                    },
                  ),
                ],
              );
            },
          ),
        ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeaderMetric(String title, String val, Color color) {
    return Column(
      children: [
        Text(
          title,
          style: const TextStyle(
            color: AppTheme.textMuted,
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.8,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          val,
          style: TextStyle(
            color: color,
            fontSize: 15,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }

  Widget _buildRealFragmentCard(StoredNodeFragment frag) {
    final shortId = frag.fragmentId.length >= 8
        ? frag.fragmentId.substring(0, 8).toUpperCase()
        : frag.fragmentId;
    final expiresStr = frag.expiresAt.toString().split(' ').first;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  const Icon(Icons.psychology_rounded,
                      size: 18, color: AppTheme.primaryLight),
                  const SizedBox(width: 8),
                  Text(
                    'FRAG-$shortId #seq${frag.sequenceNumber}',
                    style: const TextStyle(
                      color: AppTheme.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      fontFamily: 'monospace',
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppTheme.surfaceLight,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: AppTheme.border),
                ),
                child: Text(
                  '${frag.sizeBytes} B',
                  style: const TextStyle(
                    color: AppTheme.textSecondary,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),

          // Human Storage Indicator (ZERO PLAINTEXT ON DISK OR SCREEN)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: AppTheme.background,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: AppTheme.primary.withValues(alpha: 0.3),
              ),
            ),
            child: const Row(
              children: [
                Icon(
                  Icons.lock_outline_rounded,
                  size: 16,
                  color: AppTheme.secondary,
                ),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Stored in Human Memory',
                    style: TextStyle(
                      color: AppTheme.textPrimary,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Text(
                  'Zero disk plaintext',
                  style: TextStyle(
                    color: AppTheme.textMuted,
                    fontSize: 10,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  'Expires: $expiresStr',
                  style: const TextStyle(
                    color: AppTheme.textMuted,
                    fontSize: 11,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),
              const Text(
                'Status: MEMORIZED (HOSTED)',
                style: TextStyle(
                  color: AppTheme.secondary,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/mock_data.dart';
import '../models/node.dart';
import '../models/pending_delivery_fragment.dart';
import '../models/pending_recall_fragment.dart';
import '../models/stored_node_fragment.dart';
import '../services/node_service.dart';
import '../services/node_storage_service.dart';
import '../theme/app_theme.dart';
import '../widgets/calibration_dial.dart';
import '../widgets/max_width_container.dart';
import '../widgets/memorization_dialog.dart';
import '../widgets/recall_submission_dialog.dart';
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
    NodeService.instance.syncNode().then((result) async {
      if (!mounted) return;
      if (NodeService.instance.pendingRecallsNotifier.value.isNotEmpty) {
        _promptRecall(NodeService.instance.pendingRecallsNotifier.value.first);
        return;
      }
      if (result.pendingDeliveries.isNotEmpty) {
        final first = result.pendingDeliveries.first;
        if (!NodeStorageService.instance.isCapacityFull &&
            !(await NodeStorageService.instance.hasFragment(first.fragmentId))) {
          if (mounted) {
            _promptMemorization(first);
          }
        }
      }
    });
  }

  Future<void> _promptRecall(PendingRecallFragment recall) async {
    if (!mounted) return;
    RecallSubmissionDialog.show(
      context,
      recall: recall,
      onSuccess: () {
        final updated = List<PendingRecallFragment>.from(
            NodeService.instance.pendingRecallsNotifier.value)
          ..removeWhere((r) => r.assignmentId == recall.assignmentId);
        NodeService.instance.pendingRecallsNotifier.value = updated;

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'FRAGMENT RECALLED! HOSTED MEMORY SLOT FREED.',
              style: GoogleFonts.spaceMono(fontSize: 11),
            ),
            backgroundColor: AppTheme.signalOnline,
            duration: const Duration(seconds: 3),
          ),
        );

        if (updated.isNotEmpty && mounted) {
          _promptRecall(updated.first);
        }
      },
    );
  }

  Future<void> _promptMemorization(PendingDeliveryFragment delivery) async {
    if (!mounted) return;

    // Safety checks: do not prompt if station is at full capacity or already memorized
    if (NodeStorageService.instance.isCapacityFull) {
      debugPrint('Station is at capacity (${NodeStorageService.instance.activeHostedCount}/5); skipping memorization prompt');
      return;
    }
    if (await NodeStorageService.instance.hasFragment(delivery.fragmentId)) {
      debugPrint('Fragment ${delivery.fragmentId} already memorized; skipping prompt');
      return;
    }

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
          SnackBar(
            content: Text(
              'FRAGMENT MEMORIZED! ENTRUSTED TO BIOLOGICAL STORAGE.',
              style: GoogleFonts.spaceMono(fontSize: 11),
            ),
            backgroundColor: AppTheme.signalOnline,
            duration: const Duration(seconds: 3),
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
      final recallCount = NodeService.instance.pendingRecallsNotifier.value.length;
      final msg = result.isSuccess
          ? (recallCount > 0
              ? 'SYNC: $recallCount RECALL SIGNAL(S) PENDING'
              : (result.pendingDeliveries.isNotEmpty
                  ? 'SYNC: ${result.pendingDeliveries.length} FRAGMENT(S) AWAITING MEMORIZATION'
                  : (result.rebalancedCount > 0
                      ? 'SYNC: ${result.rebalancedCount} FRAGMENT(S) REBALANCED TO THIS STATION'
                      : 'STATION TELEMETRY UP TO DATE (${result.storedCount} MEMORIZED)')))
          : 'SYNC ERROR: ${result.error}';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(msg, style: GoogleFonts.spaceMono(fontSize: 11)),
          duration: const Duration(seconds: 2),
          backgroundColor:
              result.isSuccess ? AppTheme.signalOnline : AppTheme.signalOffline,
        ),
      );

      if (NodeService.instance.pendingRecallsNotifier.value.isNotEmpty) {
        _promptRecall(NodeService.instance.pendingRecallsNotifier.value.first);
      } else if (result.pendingDeliveries.isNotEmpty) {
        final first = result.pendingDeliveries.first;
        if (!NodeStorageService.instance.isCapacityFull &&
            !(await NodeStorageService.instance.hasFragment(first.fragmentId))) {
          if (mounted) {
            _promptMemorization(first);
          }
        }
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
                  ? 'STATION CIRCUIT STATUS: ONLINE'
                  : 'STATION CIRCUIT STATUS: OFFLINE',
              style: GoogleFonts.spaceMono(fontSize: 11),
            ),
            backgroundColor: isNowOnline ? AppTheme.signalOnline : AppTheme.signalOffline,
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('FAILED TO UPDATE STATION STATUS: $e', style: GoogleFonts.spaceMono(fontSize: 11)),
            backgroundColor: AppTheme.signalOffline,
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
        title: Text(
          'Station Telemetry',
          style: GoogleFonts.playfairDisplay(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
        actions: [
          ValueListenableBuilder<bool>(
            valueListenable: NodeService.instance.isSyncingNotifier,
            builder: (context, isSyncing, _) {
              return IconButton(
                icon: isSyncing
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: AppTheme.textPrimary,
                        ),
                      )
                    : const Icon(Icons.refresh_rounded, size: 20),
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
                  color: AppTheme.brass,
                  backgroundColor: AppTheme.surfaceElevated,
                );
              },
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: MaxWidthContainer(
                child: ValueListenableBuilder<NodeModel?>(
                  valueListenable: NodeService.instance.currentNodeNotifier,
                  builder: (context, node, _) {
                    final isOnline = node != null ? node.isOnline : user.isNodeOnline;
                    final nodeId = node != null && node.id.isNotEmpty
                        ? (node.id.length >= 8
                            ? 'HS-STATION-${node.id.substring(0, 8).toUpperCase()}'
                            : node.id)
                        : user.nodeId;
                    final reliabilityVal = node != null
                        ? (node.reliability <= 1.0
                            ? node.reliability * 100
                            : node.reliability)
                        : user.reliability;
                    final responseRateVal = node != null
                        ? '${(node.responseRate * 100).toStringAsFixed(1)}%'
                        : '99.2%';
                    final avgResponseMs = node?.avgResponseMs ?? 0;
                    final lastSeenStr =
                        node != null ? _formatLastSeen(node.lastSeen) : 'Active';

                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Node Switchboard Header Panel
                        Container(
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: AppTheme.surface,
                            borderRadius: BorderRadius.circular(2),
                            border: Border.all(
                              color: isOnline ? AppTheme.border : AppTheme.signalOffline.withValues(alpha: 0.6),
                              width: 1.0,
                            ),
                          ),
                          child: Column(
                            children: [
                              Row(
                                children: [
                                  Container(
                                    width: 32,
                                    height: 32,
                                    decoration: BoxDecoration(
                                      color: AppTheme.surfaceElevated,
                                      border: Border.all(
                                        color: isOnline ? AppTheme.signalOnline : AppTheme.signalOffline,
                                      ),
                                      borderRadius: BorderRadius.circular(2),
                                    ),
                                    child: Icon(
                                      Icons.dns_rounded,
                                      color: isOnline ? AppTheme.signalOnline : AppTheme.signalOffline,
                                      size: 16,
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          nodeId,
                                          style: GoogleFonts.spaceMono(
                                            color: AppTheme.textPrimary,
                                            fontSize: 12,
                                            fontWeight: FontWeight.w700,
                                          ),
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis,
                                        ),
                                        const SizedBox(height: 1),
                                        Text(
                                          isOnline ? 'CIRCUIT CONNECTED' : 'CIRCUIT DISCONNECTED',
                                          style: GoogleFonts.inter(
                                            color: isOnline ? AppTheme.signalOnline : AppTheme.signalOffline,
                                            fontSize: 10,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  _isToggling
                                      ? const SizedBox(
                                          width: 20,
                                          height: 20,
                                          child: CircularProgressIndicator(
                                            strokeWidth: 2,
                                            color: AppTheme.brass,
                                          ),
                                        )
                                      : Switch(
                                          value: isOnline,
                                          onChanged: (_) => _toggleNodeStatus(isOnline),
                                          activeTrackColor: AppTheme.signalOnline.withValues(alpha: 0.5),
                                          activeColor: AppTheme.signalOnline,
                                          inactiveThumbColor: AppTheme.textMuted,
                                          inactiveTrackColor: AppTheme.border,
                                        ),
                                ],
                              ),
                              const SizedBox(height: 12),
                              const Divider(thickness: 0.8),
                              const SizedBox(height: 6),
                              Row(
                                children: [
                                  Expanded(
                                    child: _buildHeaderMetric(
                                      'STATUS',
                                      isOnline ? 'ONLINE' : 'OFFLINE',
                                      isOnline ? AppTheme.signalOnline : AppTheme.signalOffline,
                                    ),
                                  ),
                                  Expanded(
                                    child: _buildHeaderMetric(
                                      'RELIABILITY',
                                      '${reliabilityVal.toStringAsFixed(1)}%',
                                      AppTheme.textPrimary,
                                    ),
                                  ),
                                  Expanded(
                                    child: _buildHeaderMetric(
                                      'LAST SEEN',
                                      lastSeenStr.toUpperCase(),
                                      AppTheme.textSecondary,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),

                        const SizedBox(height: 16),

                        // Responsive Layout: Dial & Response Stats
                        // Mobile: Stack vertically to give CalibrationDial full breathing room!
                        // Desktop: Side-by-side
                        LayoutBuilder(
                          builder: (context, constraints) {
                            final isMobile = constraints.maxWidth < 540;

                            if (isMobile) {
                              return Column(
                                children: [
                                  CalibrationDial(
                                    scorePercent: reliabilityVal,
                                    title: 'RETENTION ACCURACY',
                                  ),
                                  const SizedBox(height: 10),
                                  Row(
                                    children: [
                                      Expanded(
                                        child: StatCard(
                                          label: 'Response Time',
                                          value: '$avgResponseMs ms',
                                          icon: Icons.speed_rounded,
                                          subtitle: 'Circuit latency',
                                        ),
                                      ),
                                      const SizedBox(width: 8),
                                      Expanded(
                                        child: StatCard(
                                          label: 'Response Rate',
                                          value: responseRateVal,
                                          icon: Icons.checklist_rtl_rounded,
                                          accentColor: AppTheme.signalOnline,
                                          subtitle: 'Challenge success',
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              );
                            } else {
                              return Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Expanded(
                                    flex: 5,
                                    child: CalibrationDial(
                                      scorePercent: reliabilityVal,
                                      title: 'RETENTION ACCURACY',
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    flex: 5,
                                    child: Column(
                                      children: [
                                        StatCard(
                                          label: 'Response Time',
                                          value: '$avgResponseMs ms',
                                          icon: Icons.speed_rounded,
                                          subtitle: 'Circuit latency',
                                        ),
                                        const SizedBox(height: 8),
                                        StatCard(
                                          label: 'Response Rate',
                                          value: responseRateVal,
                                          icon: Icons.checklist_rtl_rounded,
                                          accentColor: AppTheme.signalOnline,
                                          subtitle: 'Challenge success',
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              );
                            }
                          },
                        ),

                        const SizedBox(height: 20),

                        // Pending Recalls Banner
                        ValueListenableBuilder<List<PendingRecallFragment>>(
                          valueListenable: NodeService.instance.pendingRecallsNotifier,
                          builder: (context, pendingRecalls, _) {
                            if (pendingRecalls.isEmpty) return const SizedBox.shrink();
                            return Container(
                              margin: const EdgeInsets.only(bottom: 16),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: AppTheme.surfaceElevated,
                                borderRadius: BorderRadius.circular(2),
                                border: Border.all(color: AppTheme.signalWarning, width: 1.0),
                              ),
                              child: Row(
                                children: [
                                  const Icon(
                                    Icons.output_rounded,
                                    color: AppTheme.signalWarning,
                                    size: 20,
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          '${pendingRecalls.length} RECALL SIGNAL(S) PENDING',
                                          style: GoogleFonts.inter(
                                            color: AppTheme.textPrimary,
                                            fontSize: 11.5,
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                        const SizedBox(height: 1),
                                        Text(
                                          'Memory owner requests fragment recall.',
                                          style: GoogleFonts.inter(
                                            color: AppTheme.textMuted,
                                            fontSize: 10.5,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  ElevatedButton(
                                    onPressed: () => _promptRecall(pendingRecalls.first),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: AppTheme.signalWarning,
                                      foregroundColor: Colors.black,
                                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                    ),
                                    child: Text(
                                      'RECALL',
                                      style: GoogleFonts.inter(
                                        fontSize: 10.5,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),

                        // Pending Deliveries Banner
                        ValueListenableBuilder<List<PendingDeliveryFragment>>(
                          valueListenable: NodeService.instance.pendingDeliveriesNotifier,
                          builder: (context, pending, _) {
                            if (pending.isEmpty) return const SizedBox.shrink();
                            return Container(
                              margin: const EdgeInsets.only(bottom: 16),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: AppTheme.surfaceElevated,
                                borderRadius: BorderRadius.circular(2),
                                border: Border.all(color: AppTheme.brass, width: 1.0),
                              ),
                              child: Row(
                                children: [
                                  const Icon(
                                    Icons.psychology_rounded,
                                    color: AppTheme.brass,
                                    size: 20,
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          '${pending.length} FRAGMENT(S) AWAITING MEMORIZATION',
                                          style: GoogleFonts.inter(
                                            color: AppTheme.textPrimary,
                                            fontSize: 11.5,
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                        const SizedBox(height: 1),
                                        Text(
                                          'Entrusted to your human memory.',
                                          style: GoogleFonts.inter(
                                            color: AppTheme.textMuted,
                                            fontSize: 10.5,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  ElevatedButton(
                                    onPressed: () => _promptMemorization(pending.first),
                                    style: ElevatedButton.styleFrom(
                                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                    ),
                                    child: Text(
                                      'MEMORIZE',
                                      style: GoogleFonts.inter(
                                        fontSize: 10.5,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),

                        // Memorized Fragments Ledger
                        ValueListenableBuilder<List<StoredNodeFragment>>(
                          valueListenable: NodeStorageService.instance.storedFragmentsNotifier,
                          builder: (context, storedFragments, _) {
                            final count = storedFragments.length;
                            final maxCap = NodeStorageService.maxNodeCapacity;
                            final isFull = count >= maxCap;
                            final isOverCapacity = count > maxCap;

                            return Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Wrap(
                                  alignment: WrapAlignment.spaceBetween,
                                  crossAxisAlignment: WrapCrossAlignment.center,
                                  spacing: 8,
                                  runSpacing: 4,
                                  children: [
                                    Text(
                                      'BIOLOGICAL STORAGE VAULT',
                                      style: GoogleFonts.inter(
                                        color: AppTheme.textMuted,
                                        fontSize: 11,
                                        fontWeight: FontWeight.w600,
                                        letterSpacing: 0.5,
                                      ),
                                    ),
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: AppTheme.surfaceElevated,
                                        borderRadius: BorderRadius.circular(3),
                                        border: Border.all(
                                          color: isFull ? AppTheme.signalWarning : AppTheme.border,
                                          width: 0.8,
                                        ),
                                      ),
                                      child: Text(
                                        '$count / $maxCap MEMORY SLOTS USED',
                                        style: GoogleFonts.spaceMono(
                                          color: isFull ? AppTheme.signalWarning : AppTheme.brassAccent,
                                          fontSize: 10,
                                          fontWeight: FontWeight.w700,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 10),

                                if (isFull) ...[
                                  Container(
                                    width: double.infinity,
                                    margin: const EdgeInsets.only(bottom: 10),
                                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                    decoration: BoxDecoration(
                                      color: AppTheme.signalWarning.withValues(alpha: 0.1),
                                      borderRadius: BorderRadius.circular(3),
                                      border: Border.all(
                                        color: AppTheme.signalWarning.withValues(alpha: 0.4),
                                        width: 1.0,
                                      ),
                                    ),
                                    child: Row(
                                      children: [
                                        const Icon(
                                          Icons.shield_outlined,
                                          size: 15,
                                          color: AppTheme.signalWarning,
                                        ),
                                        const SizedBox(width: 8),
                                        Expanded(
                                          child: Text(
                                            isOverCapacity
                                                ? 'STATION OVER CAPACITY ($count/$maxCap): Preserving existing hosted fragments. New assignments are paused until recalled.'
                                                : 'STATION FULL (5/5 SLOTS): Node is currently at full capacity. New assignments will route to other peer stations until fragments are recalled.',
                                            style: GoogleFonts.inter(
                                              color: AppTheme.textSecondary,
                                              fontSize: 11,
                                              height: 1.3,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],

                                if (storedFragments.isEmpty) ...[
                                  Container(
                                    width: double.infinity,
                                    padding: const EdgeInsets.all(20),
                                    decoration: BoxDecoration(
                                      color: AppTheme.surface,
                                      borderRadius: BorderRadius.circular(2),
                                      border: Border.all(color: AppTheme.border),
                                    ),
                                    child: Column(
                                      children: [
                                        const Icon(Icons.psychology_outlined,
                                            color: AppTheme.textMuted, size: 30),
                                        const SizedBox(height: 10),
                                        Text(
                                          'No Fragments Memorized Yet',
                                          style: GoogleFonts.playfairDisplay(
                                            color: AppTheme.textPrimary,
                                            fontSize: 14,
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                        const SizedBox(height: 4),
                                        Text(
                                          'When network participants store memories, this station will receive replica fragments for biological retention.',
                                          style: GoogleFonts.inter(
                                            color: AppTheme.textMuted,
                                            fontSize: 11.5,
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

                                const SizedBox(height: 20),
                                _buildDevControls(context, storedFragments.length),
                              ],
                            );
                          },
                        ),
                      ],
                    );
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _confirmAndClearLocalMemory() async {
    final count = NodeStorageService.instance.activeHostedCount;
    final shouldClear = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: AppTheme.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
          side: const BorderSide(color: AppTheme.border, width: 1),
        ),
        title: Row(
          children: [
            const Icon(Icons.delete_sweep_outlined, color: AppTheme.signalWarning, size: 20),
            const SizedBox(width: 8),
            Text(
              'CLEAR LOCAL MEMORY?',
              style: GoogleFonts.spaceMono(
                color: AppTheme.textPrimary,
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'This development action clears ONLY the local metadata cache ($count item(s)) tracked on this device.',
              style: GoogleFonts.inter(
                color: AppTheme.textSecondary,
                fontSize: 12,
                height: 1.4,
              ),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AppTheme.surfaceElevated,
                borderRadius: BorderRadius.circular(2),
                border: Border.all(color: AppTheme.border),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildBulletPoint('Remote Supabase memories & database remain untouched.'),
                  _buildBulletPoint('Your operator authentication account is preserved.'),
                  _buildBulletPoint('Station ledger resets immediately to 0 / 5 slots used.'),
                  _buildBulletPoint('Stale pending delivery queues will be cleared.'),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text(
              'CANCEL',
              style: GoogleFonts.spaceMono(
                color: AppTheme.textMuted,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.signalWarning.withValues(alpha: 0.15),
              foregroundColor: AppTheme.signalWarning,
              side: const BorderSide(color: AppTheme.signalWarning, width: 0.8),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            ),
            child: Text(
              'CONFIRM & CLEAR',
              style: GoogleFonts.spaceMono(
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );

    if (shouldClear == true && mounted) {
      await NodeStorageService.instance.clearAllMetadata();
      NodeService.instance.pendingDeliveriesNotifier.value = [];
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'LOCAL MEMORY CLEARED (0 / 5 SLOTS USED). SUPABASE DATA UNTOUCHED.',
              style: GoogleFonts.spaceMono(fontSize: 11),
            ),
            backgroundColor: AppTheme.signalOnline,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    }
  }

  Widget _buildBulletPoint(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('• ', style: GoogleFonts.spaceMono(color: AppTheme.brassAccent, fontSize: 11)),
          Expanded(
            child: Text(
              text,
              style: GoogleFonts.inter(color: AppTheme.textMuted, fontSize: 11, height: 1.3),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDevControls(BuildContext context, int fragmentCount) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(3),
        border: Border.all(color: AppTheme.border, width: 0.8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.build_outlined, size: 14, color: AppTheme.textMuted),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'STATION MAINTENANCE & TESTING',
                  style: GoogleFonts.inter(
                    color: AppTheme.textMuted,
                    fontSize: 10.5,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.5,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Development action: Clears local cached metadata from this station. Useful when resetting the test database to realign local capacity telemetry.',
            style: GoogleFonts.inter(
              color: AppTheme.textSecondary,
              fontSize: 11,
              height: 1.3,
            ),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            key: const Key('clear_local_memory_button'),
            onPressed: _confirmAndClearLocalMemory,
            icon: const Icon(Icons.cleaning_services_rounded, size: 14, color: AppTheme.signalWarning),
            label: Text(
              'CLEAR LOCAL MEMORY',
              style: GoogleFonts.spaceMono(
                color: AppTheme.signalWarning,
                fontSize: 10.5,
                fontWeight: FontWeight.w700,
              ),
            ),
            style: OutlinedButton.styleFrom(
              side: BorderSide(color: AppTheme.signalWarning.withValues(alpha: 0.5), width: 0.8),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeaderMetric(String title, String val, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Text(
          title,
          style: GoogleFonts.inter(
            color: AppTheme.textMuted,
            fontSize: 9.5,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.5,
          ),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(height: 2),
        Text(
          val,
          style: GoogleFonts.spaceMono(
            color: color,
            fontSize: 13,
            fontWeight: FontWeight.w700,
          ),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
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
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(2),
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
                  const Icon(Icons.psychology_rounded, size: 15, color: AppTheme.textPrimary),
                  const SizedBox(width: 8),
                  Text(
                    'FRAG-$shortId #SEQ${frag.sequenceNumber}',
                    style: GoogleFonts.spaceMono(
                      color: AppTheme.textPrimary,
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              Text(
                '${frag.sizeBytes} B',
                style: GoogleFonts.spaceMono(
                  color: AppTheme.textMuted,
                  fontSize: 10,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            decoration: BoxDecoration(
              color: AppTheme.background,
              borderRadius: BorderRadius.circular(2),
              border: Border.all(color: AppTheme.border),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.lock_outline_rounded,
                  size: 13,
                  color: AppTheme.signalOnline,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    'STORED IN HUMAN MEMORY',
                    style: GoogleFonts.spaceMono(
                      color: AppTheme.textPrimary,
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                Text(
                  'ZERO DISK PLAINTEXT',
                  style: GoogleFonts.spaceMono(
                    color: AppTheme.textMuted,
                    fontSize: 8.5,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'EXPIRES: $expiresStr',
                style: GoogleFonts.spaceMono(
                  color: AppTheme.textMuted,
                  fontSize: 9.5,
                ),
              ),
              Text(
                'MEMORIZED',
                style: GoogleFonts.spaceMono(
                  color: AppTheme.signalOnline,
                  fontSize: 9.5,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

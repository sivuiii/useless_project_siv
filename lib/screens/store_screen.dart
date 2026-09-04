import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../services/memory_service.dart';
import '../theme/app_theme.dart';
import '../widgets/max_width_container.dart';
import '../widgets/telegram_dispatch_slip.dart';

class StoreScreen extends StatefulWidget {
  const StoreScreen({super.key});

  @override
  State<StoreScreen> createState() => _StoreScreenState();
}

class _StoreScreenState extends State<StoreScreen> {
  final TextEditingController _controller = TextEditingController();
  bool _isStoring = false;
  bool _showConfirmation = false;
  String _storedSummary = '';
  MemoryStoreResult? _storeResult;

  @override
  void initState() {
    super.initState();
    MemoryService.instance.getActiveMemoriesCount();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _handleStoreMemory() async {
    final text = _controller.text.trim();
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Please enter memory text before dispatching.',
            style: GoogleFonts.inter(fontSize: 12),
          ),
          backgroundColor: AppTheme.signalWarning,
        ),
      );
      return;
    }

    setState(() {
      _isStoring = true;
    });

    try {
      final result = await MemoryService.instance.createMemory(text);
      if (mounted) {
        setState(() {
          _isStoring = false;
          _showConfirmation = true;
          _storeResult = result;
          _storedSummary = text;
          _controller.clear();
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isStoring = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              e.toString().replaceAll('Exception: ', '').replaceAll('StateError: ', ''),
              style: GoogleFonts.inter(fontSize: 12),
            ),
            backgroundColor: AppTheme.signalOffline,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    }
  }

  void _resetForm() {
    setState(() {
      _showConfirmation = false;
      _storedSummary = '';
      _storeResult = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Dispatch Memory',
          style: GoogleFonts.playfairDisplay(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: MaxWidthContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Memory Slot Usage Indicator
              ValueListenableBuilder<int>(
                valueListenable: MemoryService.instance.activeMemoriesCountNotifier,
                builder: (context, activeCount, _) {
                  final remaining = 5 - activeCount;
                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      color: AppTheme.surface,
                      borderRadius: BorderRadius.circular(2),
                      border: Border.all(color: AppTheme.border, width: 1),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.inventory_2_outlined, color: AppTheme.textPrimary, size: 16),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'ACTIVE MEMORY SLOTS',
                                style: GoogleFonts.inter(
                                  color: AppTheme.textPrimary,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 1),
                              Text(
                                '$activeCount of 5 occupied • Auto-expires in 6 months',
                                style: GoogleFonts.inter(
                                  color: AppTheme.textMuted,
                                  fontSize: 10,
                                ),
                              ),
                            ],
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: AppTheme.surfaceElevated,
                            border: Border.all(
                              color: remaining > 0 ? AppTheme.signalOnline : AppTheme.signalWarning,
                            ),
                            borderRadius: BorderRadius.circular(2),
                          ),
                          child: Text(
                            remaining > 0 ? '$remaining FREE' : 'FULL',
                            style: GoogleFonts.spaceMono(
                              color: remaining > 0 ? AppTheme.signalOnline : AppTheme.signalWarning,
                              fontSize: 9.5,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),

              const SizedBox(height: 14),

              if (_showConfirmation && _storeResult != null) ...[
                // Confirmation State Card
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppTheme.surface,
                    borderRadius: BorderRadius.circular(2),
                    border: Border.all(color: AppTheme.signalOnline, width: 1.2),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(6),
                            decoration: BoxDecoration(
                              color: AppTheme.surfaceElevated,
                              border: Border.all(color: AppTheme.signalOnline),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.check_rounded, color: AppTheme.signalOnline, size: 18),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Telegram Dispatched & Fragmented',
                                  style: GoogleFonts.playfairDisplay(
                                    color: AppTheme.textPrimary,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                Text(
                                  'ID: ${_storeResult!.memoryId.isNotEmpty ? (_storeResult!.memoryId.length >= 8 ? _storeResult!.memoryId.substring(0, 8).toUpperCase() : _storeResult!.memoryId) : 'MEM-SAVED'}',
                                  style: GoogleFonts.spaceMono(
                                    color: AppTheme.brassLight,
                                    fontSize: 11,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      const Divider(thickness: 0.8),
                      const SizedBox(height: 8),

                      // Safe Responsive Telemetry Stats Row (with Expanded to prevent overflow)
                      Row(
                        children: [
                          Expanded(child: _buildStatColumn('CREATED', '${_storeResult!.fragmentCount} FRAGS')),
                          Expanded(child: _buildStatColumn('ASSIGNED', '${_storeResult!.assignedReplicas} REPS')),
                          Expanded(child: _buildStatColumn('FULFILLED', '${_storeResult!.fulfilledReplicas} REPS')),
                          Expanded(child: _buildStatColumn('PENDING', '${_storeResult!.pendingReplicas} REPS')),
                        ],
                      ),
                      const SizedBox(height: 12),

                      if (_storeResult!.hasPendingReplicas) ...[
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: AppTheme.background,
                            borderRadius: BorderRadius.circular(2),
                            border: Border.all(color: AppTheme.border),
                          ),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Icon(Icons.hourglass_empty_rounded, color: AppTheme.signalWarning, size: 14),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  _storeResult!.assignedReplicas == 0
                                      ? 'Awaiting peer stations: Fragment created. Replicas will be assigned when peers join.'
                                      : '${_storeResult!.pendingReplicas} replica(s) pending: Waiting for additional peer human stations.',
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
                        const SizedBox(height: 12),
                      ],

                      Text(
                        'DISPATCHED PAYLOAD',
                        style: GoogleFonts.inter(
                          color: AppTheme.textMuted,
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.5,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppTheme.parchment,
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: Text(
                          _storedSummary,
                          style: GoogleFonts.courierPrime(
                            color: AppTheme.ink,
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      const SizedBox(height: 14),

                      if (_storeResult!.assignedNodeIds.isNotEmpty) ...[
                        Wrap(
                          spacing: 6,
                          runSpacing: 6,
                          children: _storeResult!.assignedNodeIds.map((id) {
                            final shortId = id.length >= 8 ? id.substring(0, 8).toUpperCase() : id;
                            return _buildNodeTag('STATION #$shortId', 'Assigned');
                          }).toList(),
                        ),
                      ] else ...[
                        _buildNodeTag('Pending Replicas', 'Awaiting peers'),
                      ],

                      const SizedBox(height: 16),
                      SizedBox(
                        width: double.infinity,
                        height: 42,
                        child: ElevatedButton.icon(
                          onPressed: _resetForm,
                          icon: const Icon(Icons.add_rounded, size: 16),
                          label: Text(
                            'DISPATCH ANOTHER MEMORY',
                            style: GoogleFonts.inter(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ] else ...[
                // Tactile Telegram Document Slip Form
                TelegramDispatchSlip(
                  controller: _controller,
                  isStoring: _isStoring,
                  onSubmit: _handleStoreMemory,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatColumn(String label, String val) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: GoogleFonts.inter(
            color: AppTheme.textMuted,
            fontSize: 9,
            fontWeight: FontWeight.w600,
          ),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(height: 2),
        Text(
          val,
          style: GoogleFonts.playfairDisplay(
            color: AppTheme.textPrimary,
            fontSize: 13,
            fontWeight: FontWeight.w700,
          ),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }

  Widget _buildNodeTag(String nodeName, String status) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(2),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 5,
            height: 5,
            decoration: BoxDecoration(
              color: status == 'Assigned' ? AppTheme.signalOnline : AppTheme.signalWarning,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            '$nodeName ($status)'.toUpperCase(),
            style: GoogleFonts.spaceMono(
              color: AppTheme.textSecondary,
              fontSize: 9.5,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

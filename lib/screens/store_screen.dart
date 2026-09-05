import 'dart:async';
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
  String _memoryId = '';
  int _packetsCreated = 0;
  int _packetsMemorized = 0;
  int _packetsPending = 0;
  Timer? _statusTimer;

  @override
  void initState() {
    super.initState();
    MemoryService.instance.getActiveMemoriesCount();
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
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

    // REQUIREMENT 1: Clear plaintext IMMEDIATELY from text field and RAM
    _controller.clear();

    setState(() {
      _isStoring = true;
    });

    try {
      final result = await MemoryService.instance.createMemory(text);
      if (mounted) {
        setState(() {
          _isStoring = false;
          _showConfirmation = true;
          _memoryId = result.memoryId;
          _packetsCreated = result.packetCount;
          _packetsPending = result.packetsPending;
          _packetsMemorized = result.packetsMemorized;
        });

        // Start live polling for memorization count updates
        _statusTimer?.cancel();
        _statusTimer = Timer.periodic(const Duration(seconds: 3), (_) {
          _pollMemoryPacketStatus();
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

  Future<void> _pollMemoryPacketStatus() async {
    if (!mounted || _memoryId.isEmpty) return;
    try {
      final status = await MemoryService.instance.getMemoryPacketStatus(_memoryId);
      if (!mounted) return;
      if (status.isNotEmpty) {
        setState(() {
          _packetsCreated = (status['packet_count'] as num?)?.toInt() ?? _packetsCreated;
          _packetsPending = (status['packets_pending'] as num?)?.toInt() ?? _packetsPending;
          _packetsMemorized = (status['packets_memorized'] as num?)?.toInt() ?? _packetsMemorized;
        });
      }
    } catch (e) {
      debugPrint('Error polling memory packet status: $e');
    }
  }

  void _resetForm() {
    _statusTimer?.cancel();
    setState(() {
      _showConfirmation = false;
      _memoryId = '';
      _packetsCreated = 0;
      _packetsMemorized = 0;
      _packetsPending = 0;
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

              if (_showConfirmation) ...[
                // Confirmation State Card (METADATA ONLY - ZERO PLAINTEXT)
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
                                  'MEMORY DISPATCHED',
                                  style: GoogleFonts.playfairDisplay(
                                    color: AppTheme.textPrimary,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                Text(
                                  'ID: ${_memoryId.isNotEmpty ? (_memoryId.length >= 8 ? 'MEM-${_memoryId.substring(0, 8).toUpperCase()}' : _memoryId) : 'MEM-SAVED'}',
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
                      const SizedBox(height: 14),
                      const Divider(thickness: 0.8),
                      const SizedBox(height: 10),

                      // Frozen Spec: PACKETS CREATED: X and PACKETS MEMORIZED: Y / X
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppTheme.surfaceElevated,
                          borderRadius: BorderRadius.circular(2),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'PACKETS CREATED: $_packetsCreated',
                                  style: GoogleFonts.spaceMono(
                                    color: AppTheme.textPrimary,
                                    fontSize: 13,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: AppTheme.background,
                                    borderRadius: BorderRadius.circular(2),
                                    border: Border.all(color: AppTheme.signalOnline),
                                  ),
                                  child: Text(
                                    'DISTRIBUTED',
                                    style: GoogleFonts.spaceMono(
                                      color: AppTheme.signalOnline,
                                      fontSize: 9,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'PACKETS MEMORIZED: $_packetsMemorized / $_packetsCreated',
                              style: GoogleFonts.spaceMono(
                                color: AppTheme.brassLight,
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),

                      if (_packetsPending > 0) ...[
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: AppTheme.background,
                            borderRadius: BorderRadius.circular(2),
                            border: Border.all(color: AppTheme.signalWarning.withValues(alpha: 0.6)),
                          ),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Icon(Icons.hourglass_empty_rounded, color: AppTheme.signalWarning, size: 14),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  'Awaiting more human nodes / incomplete distribution ($_packetsPending packets pending peer nodes).',
                                  style: GoogleFonts.inter(
                                    color: AppTheme.signalWarning,
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

                      const SizedBox(height: 8),
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
}

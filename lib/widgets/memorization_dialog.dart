import 'package:flutter/material.dart';

import '../models/pending_delivery_fragment.dart';
import '../services/memory_service.dart';
import '../theme/app_theme.dart';

/// Prominent modal dialog presented when a memory fragment is delivered to a human node.
/// 
/// [Human Server Philosophy]:
/// The human brain is the long-term storage node. This dialog shows the ephemeral
/// plaintext so the human can memorize it. Once the user taps MEMORIZED,
/// the plaintext is permanently deleted from device RAM and the server transport inbox.
class MemorizationDialog extends StatefulWidget {
  final PendingDeliveryFragment delivery;
  final VoidCallback onMemorized;

  const MemorizationDialog({
    super.key,
    required this.delivery,
    required this.onMemorized,
  });

  static Future<void> show(
    BuildContext context, {
    required PendingDeliveryFragment delivery,
    required VoidCallback onMemorized,
  }) {
    return showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => MemorizationDialog(
        delivery: delivery,
        onMemorized: onMemorized,
      ),
    );
  }

  @override
  State<MemorizationDialog> createState() => _MemorizationDialogState();
}

class _MemorizationDialogState extends State<MemorizationDialog> {
  bool _isConfirming = false;
  String? _errorMessage;

  Future<void> _handleMemorized() async {
    if (_isConfirming) return;
    setState(() {
      _isConfirming = true;
      _errorMessage = null;
    });

    try {
      await MemoryService.instance.confirmMemorizedFragment(widget.delivery);
      if (mounted) {
        widget.onMemorized();
        Navigator.of(context).pop();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isConfirming = false;
          _errorMessage = 'Failed to confirm memorization: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final frag = widget.delivery;
    final shortFragId = frag.fragmentId.length >= 8
        ? frag.fragmentId.substring(0, 8).toUpperCase()
        : frag.fragmentId;
    final expiresStr = frag.expiresAt.toString().split(' ').first;

    return PopScope(
      canPop: !_isConfirming,
      child: Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        child: Container(
          constraints: const BoxConstraints(maxWidth: 520),
          decoration: BoxDecoration(
            color: AppTheme.surface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: AppTheme.primary.withValues(alpha: 0.6),
              width: 1.5,
            ),
            boxShadow: [
              BoxShadow(
                color: AppTheme.primary.withValues(alpha: 0.2),
                blurRadius: 24,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Padding(
            padding: const EdgeInsets.all(22),
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Top icon & badge
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: AppTheme.primary.withValues(alpha: 0.15),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.psychology_rounded,
                          color: AppTheme.primary,
                          size: 28,
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: AppTheme.surfaceLight,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: Text(
                          'Seq #${frag.sequenceNumber}',
                          style: const TextStyle(
                            color: AppTheme.primaryLight,
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),

                  // Title
                  const Text(
                    'Memory Entrusted to You',
                    style: TextStyle(
                      color: AppTheme.textPrimary,
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 6),

                  // Explanation
                  const Text(
                    'You are serving as a Human Server node. The human brain is the permanent storage for this memory replica. Read and memorize the entrusted text below carefully.',
                    style: TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 13,
                      height: 1.4,
                    ),
                  ),
                  const SizedBox(height: 18),

                  // Ephemeral Plaintext Box
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: AppTheme.background,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: AppTheme.secondary.withValues(alpha: 0.5),
                        width: 1.2,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'FRAG-$shortFragId',
                              style: const TextStyle(
                                color: AppTheme.textMuted,
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                fontFamily: 'monospace',
                              ),
                            ),
                            Text(
                              '${frag.sizeBytes} BYTES',
                              style: const TextStyle(
                                color: AppTheme.textMuted,
                                fontSize: 10,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        SelectableText(
                          frag.payloadText,
                          style: const TextStyle(
                            color: AppTheme.textPrimary,
                            fontSize: 15,
                            fontFamily: 'monospace',
                            height: 1.5,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Security & Ephemeral Note
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceLight,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: AppTheme.border),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(
                          Icons.lock_clock_outlined,
                          size: 16,
                          color: AppTheme.warning,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Upon tapping MEMORIZED, this text is permanently erased from your device and the network transport buffer. Expiry: $expiresStr.',
                            style: const TextStyle(
                              color: AppTheme.textMuted,
                              fontSize: 11,
                              height: 1.3,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  if (_errorMessage != null) ...[
                    const SizedBox(height: 12),
                    Text(
                      _errorMessage!,
                      style: const TextStyle(
                        color: AppTheme.error,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],

                  const SizedBox(height: 20),

                  // Actions
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: _isConfirming
                              ? null
                              : () => Navigator.of(context).pop(),
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            side: const BorderSide(color: AppTheme.border),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10),
                            ),
                          ),
                          child: const Text(
                            'Memorize Later',
                            style: TextStyle(
                              color: AppTheme.textSecondary,
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        flex: 2,
                        child: ElevatedButton(
                          onPressed: _isConfirming ? null : _handleMemorized,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppTheme.primary,
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10),
                            ),
                            elevation: 2,
                          ),
                          child: _isConfirming
                              ? const SizedBox(
                                  width: 18,
                                  height: 18,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.check_circle_outline, size: 18),
                                    SizedBox(width: 6),
                                    Text(
                                      'MEMORIZED',
                                      style: TextStyle(
                                        color: Colors.white,
                                        fontSize: 14,
                                        fontWeight: FontWeight.w800,
                                        letterSpacing: 0.5,
                                      ),
                                    ),
                                  ],
                                ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

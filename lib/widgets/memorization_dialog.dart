import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

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

  String _formatError(dynamic error) {
    if (error is TimeoutException) {
      return error.message ??
          'Confirmation request timed out. The server did not respond in time.';
    }
    if (error is PostgrestException) {
      if (error.message.isNotEmpty) {
        return error.message;
      }
      return 'Database error (${error.code}): ${error.details ?? error.toString()}';
    }
    return error
        .toString()
        .replaceFirst('Exception: ', '')
        .replaceFirst('StateError: ', '')
        .replaceFirst('ArgumentError: ', '');
  }

  Future<void> _handleMemorized() async {
    if (_isConfirming) return;
    setState(() {
      _isConfirming = true;
      _errorMessage = null;
    });

    try {
      await MemoryService.instance.confirmMemorizedFragment(widget.delivery);
      if (mounted) {
        // Pop dialog first, then notify onMemorized to avoid nested dialog navigation conflicts
        Navigator.of(context).pop();
        widget.onMemorized();
      }
    } catch (e, stack) {
      debugPrint('Memorization confirmation error: $e\n$stack');
      if (mounted) {
        setState(() {
          _isConfirming = false;
          _errorMessage = _formatError(e);
        });
      }
    } finally {
      // Safety guarantee: under NO condition can _isConfirming remain true indefinitely
      if (mounted && _isConfirming) {
        setState(() {
          _isConfirming = false;
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
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: AppTheme.border, width: 1.0),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.7),
                blurRadius: 20,
                spreadRadius: 4,
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
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: AppTheme.surfaceElevated,
                          border: Border.all(color: AppTheme.border),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: const Icon(
                          Icons.psychology_rounded,
                          color: AppTheme.brassAccent,
                          size: 22,
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: AppTheme.surfaceElevated,
                          borderRadius: BorderRadius.circular(3),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: Text(
                          'SEQ #${frag.sequenceNumber}',
                          style: GoogleFonts.spaceMono(
                            color: AppTheme.brassAccent,
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),

                  Text(
                    'MEMORY ENTRUSTED TO YOUR BIOLOGICAL VAULT',
                    style: GoogleFonts.playfairDisplay(
                      color: AppTheme.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 6),

                  Text(
                    'You are serving as a station node. The human brain is the permanent biological storage device. Read and memorize the entrusted plaintext payload below.',
                    style: GoogleFonts.inter(
                      color: AppTheme.textSecondary,
                      fontSize: 12,
                      height: 1.4,
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Sender Identification Banner (Requirement 4)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceElevated,
                      borderRadius: BorderRadius.circular(3),
                      border: Border.all(color: AppTheme.brassAccent.withValues(alpha: 0.5)),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.person_outline_rounded, color: AppTheme.brassAccent, size: 16),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'MEMORY FROM: ${frag.senderUsername.toUpperCase()}',
                            style: GoogleFonts.spaceMono(
                              color: AppTheme.brassLight,
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Ephemeral Parchment Slip Box
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: AppTheme.parchment,
                      borderRadius: BorderRadius.circular(3),
                      border: Border.all(color: AppTheme.ink.withValues(alpha: 0.15), width: 1.0),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'FRAG-$shortFragId',
                              style: GoogleFonts.spaceMono(
                                color: AppTheme.ink,
                                fontSize: 10,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            Text(
                              '${frag.sizeBytes} BYTES',
                              style: GoogleFonts.spaceMono(
                                color: AppTheme.inkFaded,
                                fontSize: 10,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        SelectableText(
                          frag.payloadText,
                          style: GoogleFonts.courierPrime(
                            color: AppTheme.ink,
                            fontSize: 15,
                            height: 1.5,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),

                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceElevated,
                      borderRadius: BorderRadius.circular(2),
                      border: Border.all(color: AppTheme.border),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(
                          Icons.lock_clock_outlined,
                          size: 16,
                          color: AppTheme.signalWarning,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Upon tapping MEMORIZED, this plaintext is permanently erased from RAM and transport buffers. Expiry: $expiresStr.',
                            style: GoogleFonts.inter(
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
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                      decoration: BoxDecoration(
                        color: AppTheme.signalOffline.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(4),
                        border: Border.all(
                          color: AppTheme.signalOffline.withValues(alpha: 0.45),
                          width: 1.0,
                        ),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Icon(
                            Icons.error_outline,
                            size: 16,
                            color: AppTheme.signalOffline,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _errorMessage!,
                              style: GoogleFonts.spaceMono(
                                color: AppTheme.signalOffline,
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                height: 1.3,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],

                  const SizedBox(height: 20),

                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: _isConfirming ? null : () => Navigator.of(context).pop(),
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            side: const BorderSide(color: AppTheme.border),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                          ),
                          child: Text(
                            'MEMORIZE LATER',
                            style: GoogleFonts.inter(
                              color: AppTheme.textSecondary,
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
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
                            backgroundColor: AppTheme.surfaceElevated,
                            foregroundColor: AppTheme.brassAccent,
                            side: const BorderSide(color: AppTheme.borderSubtle, width: 1.0),
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                          ),
                          child: _isConfirming
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: AppTheme.brassAccent,
                                  ),
                                )
                              : Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(
                                      _errorMessage != null
                                          ? Icons.refresh_rounded
                                          : Icons.check_circle_outline,
                                      size: 16,
                                    ),
                                    const SizedBox(width: 6),
                                    Text(
                                      _errorMessage != null
                                          ? 'RETRY CONFIRMATION'
                                          : 'CONFIRM MEMORIZED',
                                      style: GoogleFonts.inter(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w700,
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

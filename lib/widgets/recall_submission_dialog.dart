import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/pending_recall_fragment.dart';
import '../services/memory_service.dart';
import '../theme/app_theme.dart';

/// Modal dialog presented to a human node when a memory owner initiates retrieval.
///
/// [Human Server Philosophy]:
/// The human node is requested to recall their memorized fragment plaintext from
/// biological memory and enter it. Once submitted and verified against the fragment hash,
/// the plaintext is transported temporarily to the owner, and the node's hosted
/// fragment slot is freed and forgotten.
class RecallSubmissionDialog extends StatefulWidget {
  final PendingRecallFragment recall;
  final VoidCallback onSuccess;

  const RecallSubmissionDialog({
    super.key,
    required this.recall,
    required this.onSuccess,
  });

  static Future<void> show(
    BuildContext context, {
    required PendingRecallFragment recall,
    required VoidCallback onSuccess,
  }) {
    return showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => RecallSubmissionDialog(
        recall: recall,
        onSuccess: onSuccess,
      ),
    );
  }

  @override
  State<RecallSubmissionDialog> createState() => _RecallSubmissionDialogState();
}

class _RecallSubmissionDialogState extends State<RecallSubmissionDialog> {
  final TextEditingController _textController = TextEditingController();
  bool _isSubmitting = false;
  String? _errorMessage;

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  String _formatError(dynamic error) {
    if (error is TimeoutException) {
      return error.message ?? 'Recall submission timed out. Server did not respond.';
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

  Future<void> _handleSubmit() async {
    final text = _textController.text.trim();
    if (text.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter your remembered fragment text.';
      });
      return;
    }

    if (_isSubmitting) return;

    setState(() {
      _isSubmitting = true;
      _errorMessage = null;
    });

    try {
      await MemoryService.instance.submitFragmentRecall(
        retrievalId: widget.recall.retrievalId,
        assignmentId: widget.recall.assignmentId,
        fragmentId: widget.recall.fragmentId,
        recalledText: text,
      );
      if (mounted) {
        Navigator.of(context).pop();
        widget.onSuccess();
      }
    } catch (e, stack) {
      debugPrint('Recall submission error: $e\n$stack');
      if (mounted) {
        setState(() {
          _isSubmitting = false;
          _errorMessage = _formatError(e);
        });
      }
    } finally {
      if (mounted && _isSubmitting) {
        setState(() {
          _isSubmitting = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final recall = widget.recall;
    final shortFragId = recall.fragmentId.length >= 8
        ? recall.fragmentId.substring(0, 8).toUpperCase()
        : recall.fragmentId;

    return PopScope(
      canPop: !_isSubmitting,
      child: Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        child: Container(
          constraints: const BoxConstraints(maxWidth: 520),
          decoration: BoxDecoration(
            color: AppTheme.surface,
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: AppTheme.signalWarning, width: 1.0),
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
                          border: Border.all(color: AppTheme.signalWarning),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: const Icon(
                          Icons.output_rounded,
                          color: AppTheme.signalWarning,
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
                          'SEQ #${recall.sequenceNumber}',
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
                    'RECALL SIGNAL RECEIVED',
                    style: GoogleFonts.playfairDisplay(
                      color: AppTheme.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 6),

                  Text(
                    'The memory owner has initiated retrieval. Please recall the exact fragment plaintext stored in your biological memory and enter it below.',
                    style: GoogleFonts.inter(
                      color: AppTheme.textSecondary,
                      fontSize: 12,
                      height: 1.4,
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Metadata Box
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceElevated,
                      borderRadius: BorderRadius.circular(3),
                      border: Border.all(color: AppTheme.border),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'FRAG-$shortFragId',
                          style: GoogleFonts.spaceMono(
                            color: AppTheme.textMuted,
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          '${recall.sizeBytes} BYTES EXPECTED',
                          style: GoogleFonts.spaceMono(
                            color: AppTheme.brassAccent,
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Plaintext Entry Field
                  Text(
                    'RECALLED PLAINTEXT:',
                    style: GoogleFonts.spaceMono(
                      color: AppTheme.textMuted,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _textController,
                    enabled: !_isSubmitting,
                    maxLines: 3,
                    style: GoogleFonts.courierPrime(
                      color: AppTheme.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                    decoration: InputDecoration(
                      hintText: 'Type your remembered fragment exactly...',
                      hintStyle: GoogleFonts.courierPrime(
                        color: AppTheme.textMuted,
                        fontSize: 13,
                      ),
                      filled: true,
                      fillColor: AppTheme.surfaceElevated,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(3),
                        borderSide: const BorderSide(color: AppTheme.border),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(3),
                        borderSide: const BorderSide(color: AppTheme.signalWarning),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),

                  if (_errorMessage != null) ...[
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: AppTheme.signalOffline.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(3),
                        border: Border.all(color: AppTheme.signalOffline.withValues(alpha: 0.5)),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.error_outline, color: AppTheme.signalOffline, size: 16),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _errorMessage!,
                              style: GoogleFonts.inter(
                                color: AppTheme.signalOffline,
                                fontSize: 12,
                                height: 1.3,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                  ],

                  // Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: _isSubmitting ? null : () => Navigator.of(context).pop(),
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: AppTheme.border),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(3),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          child: Text(
                            'LATER',
                            style: GoogleFonts.spaceMono(
                              color: AppTheme.textMuted,
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        flex: 2,
                        child: ElevatedButton(
                          onPressed: _isSubmitting ? null : _handleSubmit,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppTheme.signalWarning,
                            foregroundColor: Colors.black,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(3),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          child: _isSubmitting
                              ? const SizedBox(
                                  width: 18,
                                  height: 18,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    valueColor: AlwaysStoppedAnimation<Color>(Colors.black),
                                  ),
                                )
                              : Text(
                                  'SUBMIT RECALL',
                                  style: GoogleFonts.spaceMono(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w800,
                                    letterSpacing: 0.5,
                                  ),
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

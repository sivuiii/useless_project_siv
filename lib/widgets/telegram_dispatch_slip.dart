import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

/// A physical telegram document slip component for storing information into the human network.
/// Fully responsive across all mobile form factors (320dp - 480dp & desktop).
class TelegramDispatchSlip extends StatelessWidget {
  final TextEditingController controller;
  final bool isStoring;
  final VoidCallback onSubmit;

  const TelegramDispatchSlip({
    super.key,
    required this.controller,
    required this.isStoring,
    required this.onSubmit,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.parchment,
        borderRadius: BorderRadius.circular(2),
        border: Border.all(color: AppTheme.border, width: 1.2),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.4),
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header: Document title & status stamp
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: const BoxDecoration(
              color: AppTheme.parchmentDark,
              border: Border(bottom: BorderSide(color: AppTheme.border, width: 1.0)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'TELEGRAM DISPATCH FORM',
                        style: GoogleFonts.inter(
                          color: AppTheme.ink,
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.5,
                        ),
                      ),
                      const SizedBox(height: 1),
                      Text(
                        'CIRCUIT CONFIDENTIAL • HUMAN NETWORK',
                        style: GoogleFonts.spaceMono(
                          color: AppTheme.inkFaded,
                          fontSize: 8.5,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    border: Border.all(color: AppTheme.inkStamp, width: 1.2),
                    borderRadius: BorderRadius.circular(2),
                  ),
                  child: Text(
                    'PARITY: 3X',
                    style: GoogleFonts.spaceMono(
                      color: AppTheme.inkStamp,
                      fontSize: 9.5,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Message Body Field with Ruled Paper Lines
          Padding(
            padding: const EdgeInsets.all(14),
            child: ValueListenableBuilder<TextEditingValue>(
              valueListenable: controller,
              builder: (context, value, _) {
                final charCount = value.text.length;
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            'MESSAGE PAYLOAD',
                            style: GoogleFonts.inter(
                              color: AppTheme.inkFaded,
                              fontSize: 10.5,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        Text(
                          '$charCount CHARS',
                          style: GoogleFonts.spaceMono(
                            color: charCount > 1000 ? AppTheme.inkStamp : AppTheme.inkFaded,
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Container(
                      decoration: BoxDecoration(
                        color: AppTheme.parchment,
                        border: Border.all(color: AppTheme.inkFaded.withValues(alpha: 0.25)),
                      ),
                      child: TextField(
                        controller: controller,
                        maxLines: 6,
                        minLines: 4,
                        style: GoogleFonts.courierPrime(
                          color: AppTheme.ink,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          height: 1.45,
                        ),
                        cursorColor: AppTheme.inkStamp,
                        decoration: InputDecoration(
                          hintText: 'Type message to fragment and entrust to human memory...',
                          hintStyle: GoogleFonts.courierPrime(
                            color: AppTheme.inkFaded.withValues(alpha: 0.5),
                            fontSize: 13,
                          ),
                          filled: false,
                          border: InputBorder.none,
                          enabledBorder: InputBorder.none,
                          focusedBorder: InputBorder.none,
                          contentPadding: const EdgeInsets.all(12),
                        ),
                      ),
                    ),
                    const SizedBox(height: 14),
                    SizedBox(
                      width: double.infinity,
                      height: 44,
                      child: ElevatedButton(
                        onPressed: isStoring ? null : onSubmit,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.ink,
                          foregroundColor: AppTheme.parchment,
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(3),
                            side: const BorderSide(color: AppTheme.inkStamp, width: 1.0),
                          ),
                        ),
                        child: isStoring
                            ? Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const SizedBox(
                                    width: 14,
                                    height: 14,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: AppTheme.parchment,
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  Flexible(
                                    child: Text(
                                      'DISPATCHING TO NODES...',
                                      style: GoogleFonts.inter(
                                        fontSize: 11.5,
                                        fontWeight: FontWeight.w700,
                                        letterSpacing: 0.5,
                                      ),
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                ],
                              )
                            : Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Icon(Icons.send_rounded, size: 14, color: AppTheme.brassLight),
                                  const SizedBox(width: 8),
                                  Flexible(
                                    child: Text(
                                      'DISPATCH MEMORY TELEGRAM',
                                      style: GoogleFonts.inter(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w700,
                                        letterSpacing: 0.5,
                                      ),
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                ],
                              ),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

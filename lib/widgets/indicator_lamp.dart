import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

/// A physical telegraph indicator lamp with brass bezel, jewel lens effect,
/// and an engraved status plaque.
class IndicatorLamp extends StatelessWidget {
  final bool isOnline;
  final String? customLabel;
  final double size;
  final bool showPlaque;

  const IndicatorLamp({
    super.key,
    required this.isOnline,
    this.customLabel,
    this.size = 10,
    this.showPlaque = true,
  });

  @override
  Widget build(BuildContext context) {
    final lampColor = isOnline ? AppTheme.signalOnline : AppTheme.signalOffline;
    final defaultLabel = isOnline ? 'STATION ONLINE' : 'STATION OFFLINE';
    final labelText = customLabel ?? defaultLabel;

    final lampWidget = Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        border: Border.all(
          color: AppTheme.brassLight.withValues(alpha: 0.9),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: lampColor.withValues(alpha: isOnline ? 0.6 : 0.2),
            blurRadius: isOnline ? 6 : 2,
            spreadRadius: isOnline ? 1.5 : 0.5,
          ),
        ],
        gradient: RadialGradient(
          colors: [
            Colors.white.withValues(alpha: 0.9),
            lampColor,
            lampColor.withValues(alpha: 0.8),
          ],
          stops: const [0.0, 0.45, 1.0],
          center: const Alignment(-0.25, -0.25),
        ),
      ),
    );

    if (!showPlaque) return lampWidget;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(3),
        border: Border.all(color: AppTheme.borderSubtle, width: 1),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.35),
            blurRadius: 2,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          lampWidget,
          const SizedBox(width: 8),
          Text(
            labelText,
            style: GoogleFonts.spaceMono(
              color: isOnline ? AppTheme.textPrimary : AppTheme.textMuted,
              fontSize: 10,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.0,
            ),
          ),
        ],
      ),
    );
  }
}

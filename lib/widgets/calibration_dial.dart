import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

/// A precision mechanical instrument dial for measuring human node reliability.
/// Automatically adapts its radius to available parent constraints.
class CalibrationDial extends StatelessWidget {
  final double scorePercent; // 0.0 to 100.0
  final double? size;
  final String title;

  const CalibrationDial({
    super.key,
    required this.scorePercent,
    this.size,
    this.title = 'RETENTION INDEX',
  });

  String get conditionRating {
    if (scorePercent >= 95.0) return 'OPTIMAL';
    if (scorePercent >= 80.0) return 'GOOD';
    if (scorePercent >= 60.0) return 'DEGRADED';
    return 'STANDBY';
  }

  Color get conditionColor {
    if (scorePercent >= 95.0) return AppTheme.signalOnline;
    if (scorePercent >= 80.0) return AppTheme.brassLight;
    if (scorePercent >= 60.0) return AppTheme.signalWarning;
    return AppTheme.signalOffline;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(2),
        border: Border.all(color: AppTheme.border, width: 1.0),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final dialWidth = math.min(constraints.maxWidth * 0.7, size ?? 150.0);

          return Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      title.toUpperCase(),
                      style: GoogleFonts.inter(
                        color: AppTheme.textMuted,
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 0.5,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: AppTheme.surface,
                      border: Border.all(color: conditionColor.withValues(alpha: 0.5), width: 0.8),
                      borderRadius: BorderRadius.circular(2),
                    ),
                    child: Text(
                      conditionRating,
                      style: GoogleFonts.spaceMono(
                        color: conditionColor,
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              SizedBox(
                width: dialWidth,
                height: dialWidth * 0.55,
                child: CustomPaint(
                  painter: _DialPainter(
                    percent: (scorePercent.clamp(0.0, 100.0)) / 100.0,
                    accentColor: conditionColor,
                  ),
                ),
              ),
              const SizedBox(height: 4),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.baseline,
                textBaseline: TextBaseline.alphabetic,
                children: [
                  Text(
                    scorePercent.toStringAsFixed(1),
                    style: GoogleFonts.playfairDisplay(
                      color: AppTheme.textPrimary,
                      fontSize: 24,
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(width: 2),
                  Text(
                    '%',
                    style: GoogleFonts.spaceMono(
                      color: AppTheme.brass,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 1),
              Text(
                'BIOLOGICAL RECALL ACCURACY',
                style: GoogleFonts.inter(
                  color: AppTheme.textMuted,
                  fontSize: 9,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _DialPainter extends CustomPainter {
  final double percent; // 0.0 to 1.0
  final Color accentColor;

  _DialPainter({required this.percent, required this.accentColor});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.95);
    final radius = size.width * 0.44;

    const startAngle = math.pi;
    const sweepAngle = math.pi;

    final trackPaint = Paint()
      ..color = AppTheme.border
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.5;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      trackPaint,
    );

    final tickPaint = Paint()
      ..color = AppTheme.textMuted.withValues(alpha: 0.6)
      ..strokeWidth = 1.0;

    const tickCount = 8;
    for (int i = 0; i <= tickCount; i++) {
      final angle = startAngle + (sweepAngle * (i / tickCount));
      final innerR = radius - 5;
      final outerR = radius + 1;

      final p1 = Offset(center.dx + innerR * math.cos(angle), center.dy + innerR * math.sin(angle));
      final p2 = Offset(center.dx + outerR * math.cos(angle), center.dy + outerR * math.sin(angle));

      canvas.drawLine(p1, p2, tickPaint);
    }

    final activePaint = Paint()
      ..color = accentColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle * percent,
      false,
      activePaint,
    );

    final needleAngle = startAngle + (sweepAngle * percent);
    final needleLength = radius - 3;
    final needleTip = Offset(
      center.dx + needleLength * math.cos(needleAngle),
      center.dy + needleLength * math.sin(needleAngle),
    );

    final needlePaint = Paint()
      ..color = AppTheme.textPrimary
      ..strokeWidth = 1.8
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(center, needleTip, needlePaint);

    final hubPaint = Paint()..color = AppTheme.brass;
    canvas.drawCircle(center, 4, hubPaint);
  }

  @override
  bool shouldRepaint(covariant _DialPainter oldDelegate) {
    return oldDelegate.percent != percent || oldDelegate.accentColor != accentColor;
  }
}

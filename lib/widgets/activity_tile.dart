import 'package:flutter/material.dart';
import '../models/mock_data.dart';
import '../theme/app_theme.dart';

class ActivityTile extends StatelessWidget {
  final ActivityItemMock activity;

  const ActivityTile({super.key, required this.activity});

  IconData _getIcon() {
    switch (activity.type) {
      case 'store':
        return Icons.cloud_upload_rounded;
      case 'retrieve':
        return Icons.cloud_download_rounded;
      case 'hosted_verify':
        return Icons.verified_user_rounded;
      case 'credit_earned':
        return Icons.monetization_on_rounded;
      default:
        return Icons.timeline_rounded;
    }
  }

  Color _getColor() {
    switch (activity.type) {
      case 'store':
        return AppTheme.primary;
      case 'retrieve':
        return Colors.purpleAccent;
      case 'hosted_verify':
        return AppTheme.secondary;
      case 'credit_earned':
        return AppTheme.warning;
      default:
        return AppTheme.textSecondary;
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _getColor();

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6.0),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppTheme.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.border.withValues(alpha: 0.7)),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.12),
                shape: BoxShape.circle,
              ),
              child: Icon(_getIcon(), size: 18, color: color),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        activity.title,
                        style: const TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Text(
                        activity.timestamp,
                        style: const TextStyle(
                          color: AppTheme.textMuted,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 3),
                  Text(
                    activity.detail,
                    style: const TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 12,
                    ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

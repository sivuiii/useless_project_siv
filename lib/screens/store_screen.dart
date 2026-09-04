import 'package:flutter/material.dart';

import '../services/memory_service.dart';
import '../theme/app_theme.dart';
import '../widgets/max_width_container.dart';

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
        const SnackBar(
          content: Text('Please enter memory text before storing.'),
          backgroundColor: AppTheme.warning,
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
            ),
            backgroundColor: AppTheme.error,
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
        title: const Text('Store Memory'),
      ),
      body: SingleChildScrollView(
        child: MaxWidthContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Memory Slot Usage Banner
              ValueListenableBuilder<int>(
                valueListenable: MemoryService.instance.activeMemoriesCountNotifier,
                builder: (context, activeCount, _) {
                  final remaining = 5 - activeCount;
                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    decoration: BoxDecoration(
                      color: AppTheme.surface,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppTheme.border),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.storage_rounded,
                            color: AppTheme.primary, size: 20),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Active Storage Slots',
                                style: TextStyle(
                                  color: AppTheme.textPrimary,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                '$activeCount of 5 used • Auto-expires in 6 months',
                                style: const TextStyle(
                                  color: AppTheme.textMuted,
                                  fontSize: 11,
                                ),
                              ),
                            ],
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: (remaining > 0
                                    ? AppTheme.secondary
                                    : AppTheme.warning)
                                .withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            remaining > 0 ? '$remaining Slots Free' : 'Limit Reached',
                            style: TextStyle(
                              color: remaining > 0
                                  ? AppTheme.secondary
                                  : AppTheme.warning,
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),

              const SizedBox(height: 18),

              // Network Distribution Notice
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppTheme.primary.withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(12),
                  border:
                      Border.all(color: AppTheme.primary.withValues(alpha: 0.25)),
                ),
                child: const Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.hub_rounded,
                        color: AppTheme.primaryLight, size: 18),
                    SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Client-side fragmentation active: Your text is split into sequential chunks and distributed with 3x redundancy across human nodes. Plaintext is never stored whole on the server.',
                        style: TextStyle(
                          color: AppTheme.textSecondary,
                          fontSize: 12,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              if (_showConfirmation && _storeResult != null) ...[
                // Real Confirmation State
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: AppTheme.surface,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: AppTheme.secondary.withValues(alpha: 0.5),
                      width: 1.5,
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: AppTheme.secondary.withValues(alpha: 0.2),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.check_circle_rounded,
                                color: AppTheme.secondary, size: 24),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Memory Fragmented & Stored!',
                                  style: TextStyle(
                                    color: AppTheme.textPrimary,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                Text(
                                  'ID: ${_storeResult!.memoryId.isNotEmpty ? (_storeResult!.memoryId.length >= 8 ? _storeResult!.memoryId.substring(0, 8).toUpperCase() : _storeResult!.memoryId) : 'MEM-SAVED'}',
                                  style: const TextStyle(
                                    color: AppTheme.textMuted,
                                    fontSize: 12,
                                    fontFamily: 'monospace',
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      const Divider(),
                      const SizedBox(height: 12),

                      // Telemetry Stats Row
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          _buildStatColumn('CREATED', '${_storeResult!.fragmentCount} frags'),
                          _buildStatColumn('ASSIGNED', '${_storeResult!.assignedReplicas} reps'),
                          _buildStatColumn('FULFILLED', '${_storeResult!.fulfilledReplicas} reps'),
                          _buildStatColumn('PENDING', '${_storeResult!.pendingReplicas} reps'),
                        ],
                      ),
                      const SizedBox(height: 16),

                      // Single Node notice if replicas pending
                      if (_storeResult!.hasPendingReplicas) ...[
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: AppTheme.background,
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: AppTheme.border),
                          ),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Icon(Icons.hourglass_empty_rounded,
                                  color: AppTheme.warning, size: 16),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  _storeResult!.assignedReplicas == 0
                                      ? 'Awaiting peer nodes: Fragment metadata created. All ${_storeResult!.pendingReplicas} replicas remain PENDING because no other peer nodes are currently online (Single-node MVP mode).'
                                      : '${_storeResult!.pendingReplicas} replica(s) PENDING: Waiting for additional peer nodes to join for complete 3x redundancy.',
                                  style: const TextStyle(
                                    color: AppTheme.textSecondary,
                                    fontSize: 11,
                                    height: 1.3,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                      ],

                      const Text(
                        'ORIGINAL PAYLOAD (FRAGMENTED & DEPLOYED)',
                        style: TextStyle(
                          color: AppTheme.textMuted,
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppTheme.background,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: AppTheme.border),
                        ),
                        child: Text(
                          _storedSummary,
                          style: const TextStyle(
                            color: AppTheme.textPrimary,
                            fontSize: 13,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),

                      // Real Node Assignment Tags
                      if (_storeResult!.assignedNodeIds.isNotEmpty) ...[
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: _storeResult!.assignedNodeIds.map((id) {
                            final shortId = id.length >= 8 ? id.substring(0, 8).toUpperCase() : id;
                            return _buildNodeTag('Node #$shortId', 'Assigned');
                          }).toList(),
                        ),
                      ] else ...[
                        Row(
                          children: [
                            _buildNodeTag('Pending Replicas', 'Awaiting peers'),
                          ],
                        ),
                      ],

                      const SizedBox(height: 20),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton.icon(
                          onPressed: _resetForm,
                          icon: const Icon(Icons.add_rounded, size: 18),
                          label: const Text('Store Another Memory'),
                        ),
                      ),
                    ],
                  ),
                ),
              ] else ...[
                // Input Form
                ValueListenableBuilder<TextEditingValue>(
                  valueListenable: _controller,
                  builder: (context, value, child) {
                    final charCount = value.text.length;
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'MEMORY CONTENT',
                              style: TextStyle(
                                color: AppTheme.textMuted,
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 0.8,
                              ),
                            ),
                            Text(
                              '$charCount characters',
                              style: TextStyle(
                                color: charCount > 1000
                                    ? AppTheme.warning
                                    : AppTheme.textMuted,
                                fontSize: 11,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        TextField(
                          controller: _controller,
                          maxLines: 7,
                          minLines: 5,
                          style: const TextStyle(
                            color: AppTheme.textPrimary,
                            fontSize: 14,
                            height: 1.4,
                          ),
                          decoration: const InputDecoration(
                            hintText:
                                'Enter information you want the Human Server network to remember...\n\nExample: "The security pin for the server room is 9982 and the contact person on-call is Alex."',
                          ),
                        ),
                        const SizedBox(height: 24),

                        // Action Button
                        SizedBox(
                          width: double.infinity,
                          height: 48,
                          child: ElevatedButton(
                            onPressed: _isStoring ? null : _handleStoreMemory,
                            child: _isStoring
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.white,
                                    ),
                                  )
                                : const Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(Icons.cloud_upload_rounded, size: 20),
                                      SizedBox(width: 10),
                                      Text('Fragment & Store Memory'),
                                    ],
                                  ),
                          ),
                        ),
                      ],
                    );
                  },
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
          style: const TextStyle(
            color: AppTheme.textMuted,
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.8,
          ),
        ),
        const SizedBox(height: 3),
        Text(
          val,
          style: const TextStyle(
            color: AppTheme.textPrimary,
            fontSize: 14,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }

  Widget _buildNodeTag(String nodeName, String status) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: AppTheme.background,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              color: status == 'Assigned' ? AppTheme.secondary : AppTheme.warning,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            '$nodeName ($status)',
            style: const TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

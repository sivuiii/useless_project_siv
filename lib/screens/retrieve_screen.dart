import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/retrieval_status.dart';
import '../models/user_stored_memory.dart';
import '../services/memory_service.dart';
import '../theme/app_theme.dart';
import '../widgets/max_width_container.dart';
import '../widgets/reconstruction_visualizer.dart';

class RetrieveScreen extends StatefulWidget {
  const RetrieveScreen({super.key});

  @override
  State<RetrieveScreen> createState() => _RetrieveScreenState();
}

class _RetrieveScreenState extends State<RetrieveScreen> {
  final TextEditingController _promptController = TextEditingController();

  bool _isLoadingMemories = true;
  List<UserStoredMemory> _storedMemories = [];
  UserStoredMemory? _selectedMemory;

  bool _isRetrieving = false;
  String? _activeRetrievalId;
  RetrievalStatusResult? _retrievalStatus;
  Timer? _pollTimer;
  String? _reconstructedMessage;
  String? _errorMessage;
  bool _isPurging = false;

  @override
  void initState() {
    super.initState();
    _loadStoredMemories();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _loadStoredMemories() async {
    setState(() {
      _isLoadingMemories = true;
    });

    try {
      final memories = await MemoryService.instance.getUserStoredMemories();
      if (mounted) {
        setState(() {
          _storedMemories = memories;
          _isLoadingMemories = false;
        });
      }
    } catch (e) {
      debugPrint('Error loading user stored memories: $e');
      if (mounted) {
        setState(() {
          _isLoadingMemories = false;
        });
      }
    }
  }

  void _selectMemory(UserStoredMemory mem) {
    setState(() {
      _selectedMemory = mem;
      _promptController.text = mem.memoryId;
      _errorMessage = null;
    });
  }

  Future<void> _handleInitiateRetrieval() async {
    final query = _promptController.text.trim();
    if (query.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'PLEASE SELECT OR ENTER A STORED MEMORY ID.',
            style: GoogleFonts.spaceMono(fontSize: 12),
          ),
          backgroundColor: AppTheme.signalWarning,
        ),
      );
      return;
    }

    _pollTimer?.cancel();

    setState(() {
      _isRetrieving = true;
      _errorMessage = null;
      _reconstructedMessage = null;
      _retrievalStatus = null;
    });

    try {
      final retrievalId =
          await MemoryService.instance.initiateMemoryRetrieval(query);

      if (mounted) {
        setState(() {
          _activeRetrievalId = retrievalId;
          _isRetrieving = true;
        });
      }

      // Initial poll immediately
      await _pollStatus(retrievalId);

      // Set up periodic polling every 2.5 seconds
      _pollTimer = Timer.periodic(const Duration(milliseconds: 2500), (timer) {
        if (!mounted || _activeRetrievalId == null) {
          timer.cancel();
          return;
        }
        _pollStatus(_activeRetrievalId!);
      });
    } catch (e) {
      debugPrint('Error initiating memory retrieval: $e');
      if (mounted) {
        setState(() {
          _isRetrieving = false;
          _errorMessage = e.toString().replaceFirst('Exception: ', '');
        });
      }
    }
  }

  Future<void> _pollStatus(String retrievalId) async {
    try {
      final status = await MemoryService.instance
          .getRetrievalStatusAndFragments(retrievalId);
      if (!mounted) return;

      setState(() {
        _retrievalStatus = status;
      });

      if (status.fragments.isNotEmpty) {
        final reconstructed =
            MemoryService.reconstructFromResponses(status.fragments);

        setState(() {
          _reconstructedMessage = reconstructed;
        });
      }

      // If all hosting nodes have submitted recall responses
      if (status.hostingNodesCount > 0 &&
          status.recalledCount >= status.hostingNodesCount &&
          status.status != 'completed') {
        _pollTimer?.cancel();
        // Auto-complete retrieval to purge ephemeral transport from database
        _handleCompleteRetrieval(retrievalId);
      }
    } catch (e) {
      debugPrint('Error polling retrieval status: $e');
    }
  }

  Future<void> _handleCompleteRetrieval(String retrievalId) async {
    if (_isPurging) return;
    setState(() {
      _isPurging = true;
    });

    try {
      await MemoryService.instance.completeMemoryRetrieval(retrievalId);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'RETRIEVAL COMPLETE. EPHEMERAL TRANSPORT BUFFER PURGED FROM DB.',
              style: GoogleFonts.spaceMono(fontSize: 11),
            ),
            backgroundColor: AppTheme.signalOnline,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    } catch (e) {
      debugPrint('Error completing retrieval: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isPurging = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'RECALL SIGNAL & RECONSTRUCT',
          style: GoogleFonts.playfairDisplay(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.2,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: MaxWidthContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Prompt Description Banner
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppTheme.surface,
                  borderRadius: BorderRadius.circular(3),
                  border: Border.all(color: AppTheme.border, width: 1),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: AppTheme.surfaceElevated,
                        border: Border.all(color: AppTheme.border),
                        borderRadius: BorderRadius.circular(3),
                      ),
                      child: const Icon(Icons.psychology_rounded,
                          color: AppTheme.brassAccent, size: 20),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'SEMANTIC RECALL TELEGRAPH',
                            style: GoogleFonts.inter(
                              color: AppTheme.textPrimary,
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5,
                            ),
                          ),
                          const SizedBox(height: 3),
                          Text(
                            'Submit a recall query. Biological human nodes hosting entrusted fragments will receive prompts to recall and submit their fragments.',
                            style: GoogleFonts.inter(
                              color: AppTheme.textSecondary,
                              fontSize: 12,
                              height: 1.35,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              // Active Memories Selection Chips
              if (_isLoadingMemories)
                const LinearProgressIndicator(
                  minHeight: 2,
                  color: AppTheme.brassAccent,
                  backgroundColor: AppTheme.surfaceElevated,
                )
              else if (_storedMemories.isNotEmpty) ...[
                Text(
                  'YOUR STORED MEMORIES IN THE NETWORK',
                  style: GoogleFonts.inter(
                    color: AppTheme.textMuted,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _storedMemories.map((mem) {
                    final isSelected = _selectedMemory?.memoryId == mem.memoryId ||
                        _promptController.text.trim() == mem.memoryId;
                    return ActionChip(
                      onPressed: () => _selectMemory(mem),
                      backgroundColor: isSelected
                          ? AppTheme.brassAccent.withValues(alpha: 0.2)
                          : AppTheme.surfaceElevated,
                      side: BorderSide(
                        color: isSelected
                            ? AppTheme.brassAccent
                            : AppTheme.border,
                      ),
                      avatar: Icon(
                        Icons.fingerprint_rounded,
                        size: 14,
                        color: isSelected
                            ? AppTheme.brassAccent
                            : AppTheme.textMuted,
                      ),
                      label: Text(
                        'MEM-${mem.shortId} • ${mem.packetCount} PACKETS • ${mem.memorizedCount} / ${mem.packetCount} MEMORIZED',
                        style: GoogleFonts.spaceMono(
                          color: isSelected
                              ? AppTheme.brassAccent
                              : AppTheme.textPrimary,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    );
                  }).toList(),
                ),
                const SizedBox(height: 16),
              ],

              // Query Input Box
              Text(
                'TARGET MEMORY ID',
                style: GoogleFonts.inter(
                  color: AppTheme.textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _promptController,
                style: GoogleFonts.spaceMono(
                    color: AppTheme.textPrimary, fontSize: 13),
                decoration: InputDecoration(
                  hintText: 'Enter Memory UUID or tap one from above...',
                  hintStyle: GoogleFonts.inter(
                      color: AppTheme.textMuted, fontSize: 12),
                  prefixIcon: const Icon(Icons.search_rounded,
                      color: AppTheme.textMuted, size: 18),
                  suffixIcon: _promptController.text.isNotEmpty
                      ? IconButton(
                          icon: const Icon(Icons.clear_rounded, size: 16),
                          onPressed: () {
                            _promptController.clear();
                            setState(() {
                              _selectedMemory = null;
                            });
                          },
                        )
                      : null,
                ),
                onChanged: (_) => setState(() {}),
                onSubmitted: (_) => _handleInitiateRetrieval(),
              ),

              const SizedBox(height: 16),

              // Retrieve CTA Button (Frozen Spec: REQUEST RECOVERY)
              SizedBox(
                width: double.infinity,
                height: 48,
                child: ElevatedButton.icon(
                  onPressed: _isRetrieving ? null : _handleInitiateRetrieval,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.surfaceElevated,
                    foregroundColor: AppTheme.brassAccent,
                    side: const BorderSide(
                        color: AppTheme.borderSubtle, width: 1.0),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(4)),
                  ),
                  icon: _isRetrieving
                      ? const SizedBox.shrink()
                      : const Icon(Icons.downloading_rounded, size: 18),
                  label: _isRetrieving
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: AppTheme.brassAccent,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Flexible(
                              child: Text(
                                'TRANSMITTING RECALL TELEGRAPH...',
                                style: GoogleFonts.inter(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                  letterSpacing: 0.5,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        )
                      : Text(
                          'REQUEST RECOVERY',
                          style: GoogleFonts.inter(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.5,
                          ),
                        ),
                ),
              ),

              if (_errorMessage != null) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AppTheme.signalOffline.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(3),
                    border: Border.all(
                        color: AppTheme.signalOffline.withValues(alpha: 0.5)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline,
                          color: AppTheme.signalOffline, size: 16),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _errorMessage!,
                          style: GoogleFonts.inter(
                            color: AppTheme.signalOffline,
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],

              const SizedBox(height: 24),

              // Live Status & Collected Fragments
              if (_retrievalStatus != null) ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppTheme.surface,
                    borderRadius: BorderRadius.circular(3),
                    border: Border.all(color: AppTheme.border),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'RETRIEVAL STATUS: ${_retrievalStatus!.status.toUpperCase()}',
                            style: GoogleFonts.spaceMono(
                              color: AppTheme.brassAccent,
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.refresh_rounded, size: 18),
                            color: AppTheme.textMuted,
                            tooltip: 'Refresh Status',
                            onPressed: () {
                              if (_activeRetrievalId != null) {
                                _pollStatus(_activeRetrievalId!);
                              }
                            },
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: AppTheme.surfaceElevated,
                                borderRadius: BorderRadius.circular(2),
                                border: Border.all(color: AppTheme.border),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'HOSTING NODES',
                                    style: GoogleFonts.spaceMono(
                                        color: AppTheme.textMuted, fontSize: 10),
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    '${_retrievalStatus!.hostingNodesCount}',
                                    style: GoogleFonts.spaceMono(
                                      color: AppTheme.textPrimary,
                                      fontSize: 16,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: AppTheme.surfaceElevated,
                                borderRadius: BorderRadius.circular(2),
                                border: Border.all(color: AppTheme.border),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'RECALLED FRAGMENTS',
                                    style: GoogleFonts.spaceMono(
                                        color: AppTheme.textMuted, fontSize: 10),
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    '${_retrievalStatus!.recalledCount} / ${_retrievalStatus!.hostingNodesCount}',
                                    style: GoogleFonts.spaceMono(
                                      color: AppTheme.signalOnline,
                                      fontSize: 16,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 14),

                      Text(
                        'COLLECTED RECALLED FRAGMENTS (${_retrievalStatus!.fragments.length}):',
                        style: GoogleFonts.spaceMono(
                          color: AppTheme.textMuted,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 8),

                      if (_retrievalStatus!.fragments.isEmpty)
                        Text(
                          'Awaiting manual recall entries from human nodes...',
                          style: GoogleFonts.inter(
                            color: AppTheme.textSecondary,
                            fontSize: 12,
                            fontStyle: FontStyle.italic,
                          ),
                        )
                      else
                        ..._retrievalStatus!.fragments.map((frag) {
                          return Container(
                            margin: const EdgeInsets.only(bottom: 6),
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 8),
                            decoration: BoxDecoration(
                              color: AppTheme.surfaceElevated,
                              borderRadius: BorderRadius.circular(2),
                              border: Border.all(color: AppTheme.border),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: AppTheme.surface,
                                    borderRadius: BorderRadius.circular(2),
                                    border:
                                        Border.all(color: AppTheme.border),
                                  ),
                                  child: Text(
                                    'SEQ #${frag.sequenceNumber}',
                                    style: GoogleFonts.spaceMono(
                                      color: AppTheme.brassAccent,
                                      fontSize: 10,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    frag.fragmentText,
                                    style: GoogleFonts.courierPrime(
                                      color: AppTheme.textPrimary,
                                      fontSize: 12,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                                const SizedBox(width: 8),
                                const Icon(Icons.check_circle_rounded,
                                    color: AppTheme.signalOnline, size: 14),
                              ],
                            ),
                          );
                        }),
                    ],
                  ),
                ),
                const SizedBox(height: 16),
              ],

              // Reconstructed Message Result
              if (_reconstructedMessage != null) ...[
                ReconstructionVisualizer(
                  queryPrompt: _selectedMemory?.memoryId != null
                      ? 'MEM-${_selectedMemory!.shortId}'
                      : 'RECALL TARGET',
                  reconstructedContent: _reconstructedMessage!,
                  confidenceScore: _retrievalStatus != null &&
                          _retrievalStatus!.hostingNodesCount > 0
                      ? (_retrievalStatus!.recalledCount /
                              _retrievalStatus!.hostingNodesCount)
                          .clamp(0.0, 1.0)
                      : 1.0,
                  respondingNodesCount: _retrievalStatus?.recalledCount ?? 1,
                  storedDate: 'ENTRUSTED',
                  expiresDate: 'RECALLED',
                ),
                const SizedBox(height: 14),

                if (_retrievalStatus?.status != 'completed')
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: _isPurging
                          ? null
                          : () {
                              if (_activeRetrievalId != null) {
                                _handleCompleteRetrieval(_activeRetrievalId!);
                              }
                            },
                      style: OutlinedButton.styleFrom(
                        side: const BorderSide(color: AppTheme.signalOnline),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                      icon: const Icon(Icons.cleaning_services_rounded,
                          color: AppTheme.signalOnline, size: 16),
                      label: Text(
                        'PURGE EPHEMERAL TRANSPORT BUFFER FROM DB',
                        style: GoogleFonts.spaceMono(
                          color: AppTheme.signalOnline,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ),
              ] else if (!_isRetrieving && _retrievalStatus == null) ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: AppTheme.surface,
                    borderRadius: BorderRadius.circular(2),
                    border: Border.all(color: AppTheme.border),
                  ),
                  child: Column(
                    children: [
                      const Icon(Icons.saved_search_rounded,
                          color: AppTheme.textMuted, size: 32),
                      const SizedBox(height: 10),
                      Text(
                        'SELECT A STORED MEMORY TO INITIATE RETRIEVAL',
                        style: GoogleFonts.spaceMono(
                          color: AppTheme.textSecondary,
                          fontSize: 11,
                          letterSpacing: 0.5,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

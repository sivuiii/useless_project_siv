import 'package:flutter/material.dart';
import '../services/node_service.dart';
import 'home_screen.dart';
import 'node_screen.dart';
import 'profile_screen.dart';
import 'retrieve_screen.dart';
import 'store_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  void _onTabSelected(int index) {
    setState(() {
      _currentIndex = index;
    });

    if (index == 3) {
      // Node Screen entered: synchronize node telemetry and pending deliveries
      NodeService.instance.syncNode();
    }
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      HomeScreen(
        onNavigateToStore: () => _onTabSelected(1),
        onNavigateToRetrieve: () => _onTabSelected(2),
      ),
      const StoreScreen(),
      const RetrieveScreen(),
      const NodeScreen(),
      const ProfileScreen(),
    ];

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: screens,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: _onTabSelected,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.dashboard_outlined),
            activeIcon: Icon(Icons.dashboard_rounded),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.add_box_outlined),
            activeIcon: Icon(Icons.add_box_rounded),
            label: 'Store',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.search_rounded),
            activeIcon: Icon(Icons.saved_search_rounded),
            label: 'Retrieve',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.dns_outlined),
            activeIcon: Icon(Icons.dns_rounded),
            label: 'Node',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline_rounded),
            activeIcon: Icon(Icons.person_rounded),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}

//
//  FlyBrainApp.swift
//

import SwiftUI

@main
struct FlyBrainApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .statusBarHidden(true)
                .persistentSystemOverlays(.hidden)
        }
    }
}

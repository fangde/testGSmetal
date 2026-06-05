
//
//  SimpleMetalExampleView.swift
//  SimpleMetalExample
//

import SwiftUI
import Metal

struct SimpleMetalExampleView: View {
    let device = MTLCreateSystemDefaultDevice()!
    
    var body: some View {
        ZStack {
            SimpleMetalView(device: device)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                Text("Hello Metal!")
                    .font(.title)
                    .foregroundColor(.white)
                    .padding()
            }
        }
    }
}

#Preview {
    SimpleMetalExampleView()
}


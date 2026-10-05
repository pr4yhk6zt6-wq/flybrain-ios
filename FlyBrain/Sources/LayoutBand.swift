//
//  LayoutBand.swift
//  Where a floating pane is allowed to be.
//
//  Both screens in this app put bars over a 3-D view and let a small window
//  float on top of it: the Map screen floats the camera window (the fly's
//  eyes), the Body screen floats the two eye panes. In both cases the window
//  is draggable, and in both cases the first version of it was clamped with a
//  guessed inset — a fixed 150 points at the top, a fixed 160 at the bottom —
//  which is right on the phone it was written on and wrong on every other
//  screen, and lets a pane be dropped onto a readout or a button.
//
//  So the bars report their own heights through a preference key, the band
//  between them is computed from those heights as values, and a pane is
//  clamped into the band. Nothing here is a guess, and a pane that is outside
//  the band after a rotation or a text-size change puts itself back inside on
//  the next layout pass.
//

import SwiftUI

/// What a floating pane is allowed to occupy, in the screen's own coordinates
/// (offsets from the centre of the view it sits in):
///
/// * `x` within its own half when the panes are split left and right, so two
///   panes cannot be dragged onto each other; the whole width when there is
///   only one.
/// * `y` strictly between the two bars, so a pane cannot be dropped onto a
///   readout at the top or onto the transport at the bottom.
struct PaneBand: Equatable {
    var halfWidth: CGFloat
    /// The x a pane may not cross towards the middle. Zero when the pane is
    /// not confined to a side.
    var edge: CGFloat
    var top: CGFloat
    var bottom: CGFloat

    func clamp(_ o: CGSize, side: Double = 0) -> CGSize {
        let x: CGFloat
        if side < 0 {
            x = min(-edge, max(-halfWidth, o.width))      // the left pane
        } else if side > 0 {
            x = max(edge, min(halfWidth, o.width))        // the right pane
        } else {
            x = min(halfWidth, max(-halfWidth, o.width))  // one pane, either side
        }
        let y = bottom > top ? min(bottom, max(top, o.height)) : o.height
        return CGSize(width: x, height: y)
    }

    /// The box between two bars, for a pane of a known size.
    ///
    /// `split` is for the Body screen, where the left pane's right edge stops
    /// at the middle and the right pane's left edge stops at the same line.
    static func between(bars: BarHeight,
                        within size: CGSize,
                        pane: CGSize,
                        gap: CGFloat,
                        split: Bool,
                        inset: CGFloat = 14) -> PaneBand {
        let halfPane = pane.width / 2
        let limit = size.width / 2 - halfPane - inset
        return PaneBand(
            halfWidth: split ? max(halfPane, limit) : max(40, limit),
            edge: split ? halfPane : 0,
            top: -size.height / 2 + bars.top + pane.height / 2 + gap,
            bottom: size.height / 2 - bars.bottom - pane.height / 2 - gap)
    }
}

/// The height of whichever bar is reporting, so a pane can stay clear of both
/// without either bar guessing how tall the other is.
struct BarHeight: Equatable {
    var top: CGFloat = 0
    var bottom: CGFloat = 0
}

struct BarHeightKey: PreferenceKey {
    static let defaultValue = BarHeight()
    static func reduce(value: inout BarHeight, nextValue: () -> BarHeight) {
        let n = nextValue()
        value = BarHeight(top: max(value.top, n.top),
                          bottom: max(value.bottom, n.bottom))
    }
}

extension View {
    /// Report this view's height as the top or the bottom bar's.
    func reportBarHeight(top: Bool) -> some View {
        background(GeometryReader { g in
            Color.clear.preference(key: BarHeightKey.self,
                                   value: BarHeight(top: top ? g.size.height : 0,
                                                    bottom: top ? 0 : g.size.height))
        })
    }
}

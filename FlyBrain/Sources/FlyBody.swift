//
//  FlyBody.swift
//  The body: 6 legs, 2 wings, driven by real BANC motor neuron groups.
//
//  WHAT IS MEASURED AND WHAT IS NOT
//
//  Measured (BANC v888): which neurons are motor neurons, which muscle group
//  each innervates (front/middle/hind leg, wing power/steering/tension, neck,
//  haltere, jump), which side of the animal they are on, and the complete
//  synaptic path from photoreceptor through optic lobe, descending neurons and
//  ventral nerve cord to those motor neurons.
//
//  Modelled (by us): the muscle itself. A connectome records wiring, not force,
//  so "wing power motor neurons at 120 Hz" becomes thrust through a constant we
//  chose. Differential left/right wing drive producing yaw is real fly
//  biomechanics, but the gain is ours. The tripod gait is imposed: real flies
//  generate it in the cord and those circuits are in BANC, but a 1 ms LIF
//  network does not spontaneously produce a clean gait.
//
//  All the invented constants sit together below, labelled, so the line between
//  measurement and model stays visible.
//

import Foundation
import simd

struct FlyPose {
    var position = SIMD3<Float>(0, 0.35, 0)
    var heading: Float = 0          // yaw, radians
    var pitch: Float = 0
    var roll: Float = 0
    var velocity = SIMD3<Float>.zero

    /// 0 = standing, 1 = flying.
    var airborne: Float = 0
    /// Wing stroke phase. Real flies beat at ~200 Hz; we draw the envelope
    /// rather than every stroke, or it is just a blur.
    var wingPhase: Float = 0
    var wingAmplitude: Float = 0
    var wingAsymmetry: Float = 0    // + = right beats harder, fly yaws left
    /// Tripod gait phase.
    var gaitPhase: Float = 0
    var proboscisExtension: Float = 0
    var headYaw: Float = 0
}

struct FlyDrives {
    var wingPowerL: Float = 0
    var wingPowerR: Float = 0
    var wingSteerL: Float = 0
    var wingSteerR: Float = 0
    var legL: Float = 0
    var legR: Float = 0
    var neck: Float = 0
    var proboscis: Float = 0
    var jump: Float = 0
    var haltere: Float = 0
}

final class FlyBody {

    private(set) var pose = FlyPose()
    private(set) var drives = FlyDrives()
    private(set) var energy: Float = 1.0
    private(set) var isEating = false
    private(set) var bumped = false
    private(set) var hurt: Float = 0

    // ---- the modelled muscle gains ----------------------------------------
    private let threshold: Float = 2.0          // Hz below which a muscle idles
    private let thrustPerHz: Float = 0.0016
    private let yawPerHz: Float = 0.010
    private let walkPerHz: Float = 0.0022
    private let turnPerHz: Float = 0.0075
    private let liftThresholdHz: Float = 25.0

    func reset(at p: SIMD3<Float> = SIMD3<Float>(0, 0.35, 0)) {
        pose = FlyPose(position: p)
        energy = 1.0
        hurt = 0
    }

    /// Pull the firing rate of every motor group out of the simulation.
    func readMotorDrives(from sim: SimulationEngine) {
        drives.wingPowerL = sim.groupRate("motor_wing_power_left")
        drives.wingPowerR = sim.groupRate("motor_wing_power_right")
        drives.wingSteerL = sim.groupRate("motor_wing_steering_left")
        drives.wingSteerR = sim.groupRate("motor_wing_steering_right")
        drives.legL = (sim.groupRate("motor_front_leg_left")
                     + sim.groupRate("motor_middle_leg_left")
                     + sim.groupRate("motor_hind_leg_left")) / 3
        drives.legR = (sim.groupRate("motor_front_leg_right")
                     + sim.groupRate("motor_middle_leg_right")
                     + sim.groupRate("motor_hind_leg_right")) / 3
        drives.neck = sim.groupRate("motor_neck")
        drives.proboscis = sim.groupRate("motor_proboscis")
        drives.jump = sim.groupRate("motor_jump_escape")
        drives.haltere = sim.groupRate("motor_haltere")
    }

    func update(dt: Float, world: World) {
        bumped = false
        hurt = max(0, hurt - dt * 1.5)

        let powerMean = (drives.wingPowerL + drives.wingPowerR) * 0.5
        let steerDiff = drives.wingSteerR - drives.wingSteerL
        let legMean = (drives.legL + drives.legR) * 0.5
        let legDiff = drives.legR - drives.legL

        // ---- flight or walking ---------------------------------------------
        let wantsFlight = powerMean > liftThresholdHz
        pose.airborne += ((wantsFlight ? 1 : 0) - pose.airborne) * min(1, dt * 3.5)

        pose.wingAmplitude += (min(powerMean / 120.0, 1.3) - pose.wingAmplitude) * min(1, dt * 8)
        pose.wingAsymmetry = max(-1, min(1, steerDiff / 60.0))
        pose.wingPhase += dt * (18.0 + pose.wingAmplitude * 26.0)

        // The giant-fibre escape. Two neurons, and when they go the fly is gone.
        if drives.jump > 8 && pose.position.y < 0.6 {
            pose.velocity.y += 4.5
            pose.airborne = 1
        }

        if pose.airborne > 0.5 {
            let thrust = max(0, powerMean - liftThresholdHz) * thrustPerHz
            let fwd = SIMD3<Float>(sin(pose.heading), 0, -cos(pose.heading))
            pose.heading -= pose.wingAsymmetry * yawPerHz * 60 * dt
            pose.velocity += fwd * thrust * 60 * dt
            let lift = (powerMean / 110.0) * 9.8 * 0.22
            pose.velocity.y += (lift - 9.8 * 0.22) * dt
            pose.velocity *= (1 - 1.8 * dt)                 // air drag
            pose.pitch = -min(0.5, length(SIMD2<Float>(pose.velocity.x, pose.velocity.z)) * 0.3)
            pose.roll += (pose.wingAsymmetry * 0.6 - pose.roll) * min(1, dt * 5)
            pose.gaitPhase += dt * 2
        } else {
            let speed = max(0, legMean - threshold) * walkPerHz
            pose.heading -= legDiff * turnPerHz * dt * 60
            let fwd = SIMD3<Float>(sin(pose.heading), 0, -cos(pose.heading))
            pose.velocity = fwd * speed * 60
            pose.velocity.y = min(pose.velocity.y, 0) - 9.8 * 0.25 * dt
            pose.gaitPhase += dt * (3.0 + speed * 90)
            pose.roll += (0 - pose.roll) * min(1, dt * 6)
            pose.pitch += (0 - pose.pitch) * min(1, dt * 6)
        }

        pose.position += pose.velocity * dt

        // ---- the world pushes back -------------------------------------------
        let radius: Float = 0.12
        if let normal = world.collision(at: pose.position, radius: radius) {
            bumped = true
            pose.position += normal * radius * 0.6
            let into = dot(pose.velocity, normal)
            if into < 0 { pose.velocity -= normal * into }
            if normal.y > 0.7 {
                pose.position.y = max(pose.position.y, radius)
                pose.velocity.y = max(0, pose.velocity.y)
            }
        }
        let b = world.bounds
        if abs(pose.position.x) > b {
            pose.position.x = max(-b, min(b, pose.position.x))
            pose.velocity.x *= -0.4
            bumped = true
        }
        if abs(pose.position.z) > b {
            pose.position.z = max(-b, min(b, pose.position.z))
            pose.velocity.z *= -0.4
            bumped = true
        }
        if pose.position.y > 4.0 {
            pose.position.y = 4.0
            pose.velocity.y = min(0, pose.velocity.y)
        }

        // ---- the swatter -------------------------------------------------------
        if let s = world.swatter, world.swatImpact {
            let d = SIMD2<Float>(pose.position.x - s.position.x,
                                 pose.position.z - s.position.z)
            if length(d) < s.size.x * 0.5 && pose.position.y < s.position.y + 0.25 {
                hurt = 1.0
            }
        }

        // ---- eating --------------------------------------------------------------
        isEating = false
        if let food = world.nearestFood(to: pose.position),
           length(food.position - pose.position) < 0.3,
           pose.airborne < 0.4 {
            isEating = drives.proboscis > 1.5
            if isEating {
                world.consume(food.id, amount: dt * 0.25)
                energy = min(1.2, energy + dt * 0.3)
            }
        }
        pose.proboscisExtension += ((isEating ? 1 : 0) - pose.proboscisExtension) * min(1, dt * 8)
        pose.headYaw += (max(-0.6, min(0.6, drives.neck / 40.0 - 0.3)) - pose.headYaw) * min(1, dt * 4)
        energy = max(0, energy - dt * 0.012)
    }

    /// Where the eyes are, and where they look. The fly-eye camera renders from
    /// here, and that image is what the photoreceptors see.
    var eyeTransform: (position: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>) {
        let yaw = pose.heading + pose.headYaw
        let f = SIMD3<Float>(sin(yaw) * cos(pose.pitch),
                             sin(pose.pitch),
                             -cos(yaw) * cos(pose.pitch))
        let headOffset = SIMD3<Float>(sin(yaw), 0.04, -cos(yaw)) * 0.1
        return (pose.position + headOffset, normalize(f), SIMD3<Float>(0, 1, 0))
    }

    /// Push the world's state back into the nervous system.
    func writeSensoryDrives(to sim: SimulationEngine, world: World, visionActive: Bool) {
        let (odourStrength, odourBias) = world.odour(at: pose.position, heading: pose.heading)

        // Smell. The antennal lobe gets concentration; the lateral bias is
        // folded in so the two sides are not identical.
        sim.setGroupDrive("sensory_olfactory", min(2.2, odourStrength * 0.9))
        sim.setGroupDrive("sensory_antenna", min(2.0, odourStrength * 0.6
                                                 + abs(odourBias) * 0.4))

        // Taste, only while actually on food.
        sim.setGroupDrive("sensory_gustatory", isEating ? 1.8 : 0)

        // Touch, from contact with the ground or an obstacle.
        let contact: Float = pose.airborne < 0.5 ? 0.6 : 0
        sim.setGroupDrive("sensory_tactile", contact + (bumped ? 1.4 : 0))

        // Proprioception follows the gait and the wing beat — the feedback that
        // keeps the cord informed about its own limbs.
        let gait = 0.5 + 0.5 * sin(pose.gaitPhase)
        sim.setGroupDrive("sensory_proprioception",
                          0.4 + gait * 0.5 + pose.wingAmplitude * 0.6)
        let grounded = pose.airborne < 0.5
        sim.setGroupDrive("sensory_front_leg", grounded ? 0.3 + gait * 0.6 : 0)
        sim.setGroupDrive("sensory_middle_leg", grounded ? 0.3 + (1 - gait) * 0.6 : 0)
        sim.setGroupDrive("sensory_hind_leg", grounded ? 0.3 + gait * 0.6 : 0)

        // Halteres are gyroscopes: they report rotation during flight.
        sim.setGroupDrive("sensory_haltere", pose.airborne * (0.5 + abs(pose.wingAsymmetry)))
        sim.setGroupDrive("sensory_wing", pose.wingAmplitude * 0.8)

        // Pain. The looming swatter drives nociception hard, which is what the
        // escape pathway is listening for.
        sim.setGroupDrive("sensory_nociception", hurt * 3.0)

        // Vision is written by the retina kernel from the fly-eye texture, so
        // hand the group back rather than overwriting it.
        sim.setGroupDrive("sensory_vision", visionActive ? nil : 0.4)
    }
}

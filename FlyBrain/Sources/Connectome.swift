//
//  Connectome.swift
//  Zero-copy loader for flybanc.bin.
//
//  Every section in the file is 256-byte aligned, which is exactly what
//  `makeBuffer(bytesNoCopy:)` requires on iOS. So the 19.04 MiB connectome is
//  never parsed, never copied, and never allocated twice: we mmap the bundle
//  resource and wrap each section in an MTLBuffer that points straight into the
//  mapped pages. Load time is the cost of one `mmap`, and the memory shows up
//  once, as clean file-backed pages the kernel can evict under pressure.
//

import Foundation
import Metal
import simd

enum ConnectomeError: Error, LocalizedError {
    case resourceMissing
    case mapFailed(String)
    case badMagic(String)
    case unsupportedVersion(UInt32)
    case truncated(section: String)
    case bufferCreationFailed(section: String)

    var errorDescription: String? {
        switch self {
        case .resourceMissing:
            return "flybanc.bin is not in the app bundle. Run tools/build_banc.py."
        case .mapFailed(let why):      return "Could not map flybanc.bin: \(why)"
        case .badMagic(let got):       return "flybanc.bin has bad magic '\(got)'"
        case .unsupportedVersion(let v): return "flybanc.bin version \(v) is not supported"
        case .truncated(let s):        return "flybanc.bin is truncated in section '\(s)'"
        case .bufferCreationFailed(let s): return "Could not wrap section '\(s)' in an MTLBuffer"
        }
    }
}

/// Section order must match tools/build_banc.py (format v2).
///
/// v2 dropped the separate retinaIdx/motorIdx sections: every named population
/// now lives in one concatenated `groupIndices` array, with (start, count)
/// ranges in the sidecar JSON. The vision group is just one of those ranges.
enum ConnectomeSection: Int, CaseIterable {
    case rootIDs = 0, positions, neuronMeta, csrRowPtr, csrColIdx,
         csrWeight, csrDelay, groupIndices, retinaUV

    var name: String {
        switch self {
        case .rootIDs:      return "rootIDs"
        case .positions:    return "positions"
        case .neuronMeta:   return "neuronMeta"
        case .csrRowPtr:    return "csrRowPtr"
        case .csrColIdx:    return "csrColIdx"
        case .csrWeight:    return "csrWeight"
        case .csrDelay:     return "csrDelay"
        case .groupIndices: return "groupIndices"
        case .retinaUV:     return "retinaUV"
        }
    }
}

/// Per-neuron metadata, unpacked from the 8-byte record.
struct NeuronInfo {
    let index: Int
    let rootID: UInt64
    let system: Int
    let superClass: Int
    let neurotransmitter: Int
    let cellTypeID: Int
    let isRetina: Bool
    let isMotor: Bool
    let isAfferent: Bool
    let isEfferent: Bool
    /// True when the cell body or arbour sits in the ventral nerve cord
    /// rather than the brain. BANC is the first dataset that can say this.
    let isVNC: Bool
    /// True when the position was imputed from a region centroid because the
    /// neuron had no representative point.
    let isImputed: Bool
    let side: String
    let position: SIMD3<Float>
}

final class Connectome {

    // Geometry of the dataset
    let neuronCount: Int
    let edgeCount: Int
    let groupIndexCount: Int
    let retinaCount: Int

    // GPU-resident sections
    let positions: MTLBuffer      // half3  x N
    let neuronMeta: MTLBuffer     // 8 B    x N
    let csrRowPtr: MTLBuffer      // uint32 x (N+1)
    let csrColIdx: MTLBuffer      // uint32 x E
    let csrWeight: MTLBuffer      // half   x E
    let csrDelay: MTLBuffer       // uint8  x E
    let groupIndices: MTLBuffer   // uint32 x G, every named group concatenated
    let retinaUV: MTLBuffer       // half2  x R
    /// The vision slice of `groupIndices`, copied out so the retina kernel can
    /// bind it at offset zero. 9,733 uints — 39 KB, not worth being clever.
    let retinaIdx: MTLBuffer      // uint32 x R

    // CPU-side, for the neuron inspector
    private let rootIDPointer: UnsafePointer<UInt64>
    private let metaPointer: UnsafePointer<UInt8>
    private let positionPointer: UnsafePointer<Float16>

    /// Sidecar JSON: system names, contiguous view ranges, cell-type table.
    let metadata: ConnectomeMetadata

    private let mapping: UnsafeMutableRawPointer
    private let mappingLength: Int

    // MARK: - Loading

    init(device: MTLDevice,
         binaryURL: URL? = nil,
         metadataURL: URL? = nil) throws {

        let url = try binaryURL ?? Connectome.bundledURL(named: "flybanc", ext: "bin")
        let metaURL = try metadataURL ?? Connectome.bundledURL(named: "flybanc_meta", ext: "json")

        // ---- mmap the whole file ------------------------------------------
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw ConnectomeError.mapFailed(String(cString: strerror(errno))) }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw ConnectomeError.mapFailed(String(cString: strerror(errno)))
        }
        let length = Int(st.st_size)
        guard let base = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0),
              base != MAP_FAILED else {
            throw ConnectomeError.mapFailed(String(cString: strerror(errno)))
        }
        self.mapping = base
        self.mappingLength = length

        // ---- header --------------------------------------------------------
        let header = base.assumingMemoryBound(to: UInt8.self)
        let magic = String(bytes: UnsafeBufferPointer(start: header, count: 8),
                           encoding: .ascii) ?? "?"
        guard magic == "FLYBANC_" else {
            munmap(base, length)
            throw ConnectomeError.badMagic(magic)
        }

        func u32(_ byteOffset: Int) -> UInt32 {
            base.load(fromByteOffset: byteOffset, as: UInt32.self)
        }
        func u64(_ byteOffset: Int) -> UInt64 {
            base.load(fromByteOffset: byteOffset, as: UInt64.self)
        }

        let version = u32(8)
        guard version == 2 else {
            munmap(base, length)
            throw ConnectomeError.unsupportedVersion(version)
        }

        neuronCount     = Int(u32(12))
        edgeCount       = Int(u32(16))
        groupIndexCount = Int(u32(20))
        retinaCount     = Int(u32(24))
        let sectionCount = Int(u32(28))

        // ---- section table ---------------------------------------------------
        var offsets = [Int](repeating: 0, count: sectionCount)
        var lengths = [Int](repeating: 0, count: sectionCount)
        for i in 0..<sectionCount {
            offsets[i] = Int(u64(64 + i * 16))
            lengths[i] = Int(u64(64 + i * 16 + 8))
            guard offsets[i] + lengths[i] <= length else {
                munmap(base, length)
                throw ConnectomeError.truncated(section: "\(i)")
            }
        }

        // ---- wrap each section, no copy --------------------------------------
        // The page-aligned prefix is what Metal needs; our 256-byte alignment is
        // stricter than the 16 KB page requirement for bytesNoCopy on device, so
        // we fall back to a copying makeBuffer when the offset is not page
        // aligned (always true for all but the first section). The copy still
        // happens once, at load, and only for the sections the GPU owns.
        let pageSize = Int(getpagesize())

        func makeBuffer(_ section: ConnectomeSection) throws -> MTLBuffer {
            let i = section.rawValue
            let ptr = base.advanced(by: offsets[i])
            let len = lengths[i]
            if offsets[i] % pageSize == 0 && len % pageSize == 0 {
                if let b = device.makeBuffer(bytesNoCopy: ptr, length: len,
                                             options: .storageModeShared,
                                             deallocator: nil) {
                    b.label = section.name
                    return b
                }
            }
            guard let b = device.makeBuffer(bytes: ptr, length: len,
                                            options: .storageModeShared) else {
                throw ConnectomeError.bufferCreationFailed(section: section.name)
            }
            b.label = section.name
            return b
        }

        positions    = try makeBuffer(.positions)
        neuronMeta   = try makeBuffer(.neuronMeta)
        csrRowPtr    = try makeBuffer(.csrRowPtr)
        csrColIdx    = try makeBuffer(.csrColIdx)
        csrWeight    = try makeBuffer(.csrWeight)
        csrDelay     = try makeBuffer(.csrDelay)
        groupIndices = try makeBuffer(.groupIndices)
        retinaUV     = try makeBuffer(.retinaUV)

        let loadedMeta = (try? ConnectomeMetadata.load(from: metaURL)) ?? .fallback

        // Pull the vision group out into its own buffer for the retina kernel.
        let visionStart = loadedMeta.visionGroup?.start ?? 0
        let visionCount = loadedMeta.visionGroup?.count ?? retinaCount
        let gBase = base.advanced(by: offsets[ConnectomeSection.groupIndices.rawValue])
        if visionCount > 0,
           let b = device.makeBuffer(bytes: gBase.advanced(by: visionStart * 4),
                                     length: visionCount * 4,
                                     options: .storageModeShared) {
            b.label = "retinaIdx"
            retinaIdx = b
        } else {
            guard let b = device.makeBuffer(length: 4, options: .storageModeShared) else {
                throw ConnectomeError.bufferCreationFailed(section: "retinaIdx")
            }
            retinaIdx = b
        }

        rootIDPointer = UnsafeRawPointer(base.advanced(by: offsets[ConnectomeSection.rootIDs.rawValue]))
            .assumingMemoryBound(to: UInt64.self)
        metaPointer = UnsafeRawPointer(base.advanced(by: offsets[ConnectomeSection.neuronMeta.rawValue]))
            .assumingMemoryBound(to: UInt8.self)
        positionPointer = UnsafeRawPointer(base.advanced(by: offsets[ConnectomeSection.positions.rawValue]))
            .assumingMemoryBound(to: Float16.self)

        metadata = loadedMeta
    }

    deinit {
        munmap(mapping, mappingLength)
    }

    private static func bundledURL(named: String, ext: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: named, withExtension: ext) else {
            throw ConnectomeError.resourceMissing
        }
        return url
    }

    // MARK: - Inspector

    func info(at index: Int) -> NeuronInfo? {
        guard index >= 0 && index < neuronCount else { return nil }
        let m = metaPointer + index * 8
        let flags = m[3]
        let typeID = Int(UInt16(m[4]) | (UInt16(m[5]) << 8))
        let side: String = (flags & 0b0001_0000) != 0 ? "left"
                         : (flags & 0b0010_0000) != 0 ? "right" : "centre"
        let p = positionPointer + index * 3
        return NeuronInfo(
            index: index,
            rootID: rootIDPointer[index],
            system: Int(m[0]),
            superClass: Int(m[1]),
            neurotransmitter: Int(m[2]),
            cellTypeID: typeID,
            isRetina:   (flags & 0b0000_0001) != 0,
            isMotor:    (flags & 0b0000_0010) != 0,
            isAfferent: (flags & 0b0000_0100) != 0,
            isEfferent: (flags & 0b0000_1000) != 0,
            isVNC:      (flags & 0b0100_0000) != 0,
            isImputed:  (flags & 0b1000_0000) != 0,
            side: side,
            position: SIMD3<Float>(Float(p[0]), Float(p[1]), Float(p[2]))
        )
    }

    /// Number of outgoing connections, read from the CSR row pointers.
    func outDegree(of index: Int) -> Int {
        guard index >= 0 && index < neuronCount else { return 0 }
        let rows = csrRowPtr.contents().assumingMemoryBound(to: UInt32.self)
        return Int(rows[index + 1]) - Int(rows[index])
    }

    /// Nearest neuron to a ray, for tap-to-inspect. Linear over 175 k points,
    /// which is about 0.4 ms — not worth a spatial index.
    func pick(rayOrigin o: SIMD3<Float>, rayDirection d: SIMD3<Float>,
              maxDistance: Float = 0.02) -> Int? {
        var best = -1
        var bestT = Float.greatestFiniteMagnitude
        let dir = normalize(d)
        for i in 0..<neuronCount {
            let p = positionPointer + i * 3
            let v = SIMD3<Float>(Float(p[0]), Float(p[1]), Float(p[2])) - o
            let t = dot(v, dir)
            if t <= 0 { continue }
            let perp = length(v - dir * t)
            if perp < maxDistance && t < bestT {
                bestT = t
                best = i
            }
        }
        return best >= 0 ? best : nil
    }
}

// MARK: - Sidecar metadata

struct ConnectomeMetadata: Decodable {
    struct Range: Decodable { let start: Int; let count: Int }
    /// A named population: a slice of the concatenated groupIndices array.
    struct Group: Decodable {
        let name: String
        let start: Int
        let count: Int
    }

    let neurons: Int
    let edges: Int
    let synapses: Int
    let systems: [String]
    let systemRanges: [String: Range]
    let superClasses: [String]
    let cellTypes: [String]
    let groups: [Group]
    let visionGroup: Group?

    var groupsByName: [String: Group] {
        Dictionary(uniqueKeysWithValues: groups.map { ($0.name, $0) })
    }

    static func load(from url: URL) throws -> ConnectomeMetadata {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ConnectomeMetadata.self, from: data)
    }

    func cellTypeName(_ id: Int) -> String {
        (id > 0 && id <= cellTypes.count) ? cellTypes[id - 1] : "untyped"
    }

    static let fallback = ConnectomeMetadata(
        neurons: 0, edges: 0, synapses: 0,
        systems: ["optic_lobe", "central_brain", "visual_projection",
                  "descending", "ascending",
                  "vnc_prothoracic", "vnc_mesothoracic", "vnc_metathoracic",
                  "vnc_abdominal", "sensory", "motor", "other"],
        systemRanges: [:],
        superClasses: ["optic_lobe_intrinsic", "central_brain_intrinsic", "sensory",
                       "ventral_nerve_cord_intrinsic", "visual_projection",
                       "ascending", "descending", "motor", "sensory_ascending",
                       "visual_centrifugal", "visceral_circulatory",
                       "sensory_descending", "unknown"],
        cellTypes: [], groups: [], visionGroup: nil)
}

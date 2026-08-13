package committee

import (
	"blockEmulator/params"
	"blockEmulator/utils"
	"math"
	"sort"
)

func springActionMaskCount(mask []int) int {
	count := 0
	for _, value := range mask {
		if value > 0 {
			count++
		}
	}
	return count
}

func springNormalizeActionMask(mask []int) []int {
	shards := params.ShardNum
	if shards <= 0 {
		return []int{}
	}
	normalized := make([]int, shards)
	for sid := 0; sid < shards; sid++ {
		if sid < len(mask) && mask[sid] > 0 {
			normalized[sid] = 1
		}
	}
	if springActionMaskCount(normalized) == 0 {
		for sid := 0; sid < shards; sid++ {
			normalized[sid] = 1
		}
	}
	return normalized
}

func springActionAllowed(mask []int, sid uint64) bool {
	normalized := springNormalizeActionMask(mask)
	idx := int(sid)
	return idx >= 0 && idx < len(normalized) && normalized[idx] > 0
}

func (rthm *RelayCommitteeModule) springCandidateLoadPressures() []float64 {
	shards := params.ShardNum
	pressures := make([]float64, shards)
	if rthm == nil || shards <= 0 {
		return pressures
	}

	totalPlacement := 0
	for sid := 0; sid < shards && sid < len(rthm.springShardLoad); sid++ {
		totalPlacement += rthm.springShardLoad[sid]
	}
	placementScale := float64(params.MaxBlockSize_global)
	if placementScale <= 0 {
		placementScale = 1.0
	}

	for sid := 0; sid < shards; sid++ {
		placementPressure := 0.0
		if totalPlacement > 0 && sid < len(rthm.springShardLoad) {
			placementPressure = float64(rthm.springShardLoad[sid]) /
				float64(totalPlacement) * placementScale
		}

		// Python initializes five zero windows and then appends one full
		// block-stage load vector per batch. Dividing by five keeps the online
		// and offline candidate masks on the same scale during cold start.
		recentPressure := 0.0
		for back := 0; back < 5; back++ {
			recentPressure += float64(rthm.springGetStat(uint64(sid), back).NumTx)
		}
		recentPressure /= 5.0
		pressures[sid] = placementPressure + recentPressure
	}
	return pressures
}

func (rthm *RelayCommitteeModule) springCandidateScores(
	addr utils.Address,
	senderPos []float64,
) []float64 {
	shards := params.ShardNum
	scores := make([]float64, shards)
	hashSid := uint64(utils.Addr2Shard(addr))
	loadWeight := params.SpringCandidateLoadWeight
	if loadWeight <= 0 {
		loadWeight = 1.0
	}
	pressures := rthm.springCandidateLoadPressures()
	for sid := 0; sid < shards; sid++ {
		relatedScore := 0.0
		if sid < len(senderPos) {
			relatedScore = senderPos[sid]
		}
		load := 0.0
		if sid < len(pressures) {
			load = pressures[sid]
		}
		score := relatedScore*1000.0 - loadWeight*load
		if uint64(sid) == hashSid {
			score += 0.001
		}
		scores[sid] = score
	}
	return scores
}

func (rthm *RelayCommitteeModule) springBuildCandidateActionMask(
	addr utils.Address,
	senderPos []float64,
) []int {
	shards := params.ShardNum
	if shards <= 0 {
		return []int{}
	}
	if params.SpringCandidateTopK <= 0 && params.SpringCapacityGuard == 0 {
		mask := make([]int, shards)
		for sid := range mask {
			mask[sid] = 1
		}
		return mask
	}

	mask := make([]int, shards)
	for sid := range mask {
		mask[sid] = 1
	}

	if params.SpringCapacityGuard != 0 && rthm != nil && len(rthm.springShardLoad) > 0 {
		pressures := rthm.springCandidateLoadPressures()
		totalLoad := 0.0
		for sid := 0; sid < shards && sid < len(pressures); sid++ {
			totalLoad += pressures[sid]
		}
		meanLoad := totalLoad / float64(shards)
		if meanLoad > 1e-9 {
			factor := params.SpringCapacityGuardFactor
			if factor < 1.0 {
				factor = 1.0
			}
			threshold := meanLoad * factor
			guarded := make([]int, shards)
			for sid := 0; sid < shards; sid++ {
				load := 0.0
				if sid < len(pressures) {
					load = pressures[sid]
				}
				if load <= threshold {
					guarded[sid] = 1
				}
			}
			if springActionMaskCount(guarded) > 0 {
				mask = guarded
			}
		}
	}

	topK := params.SpringCandidateTopK
	if topK > 0 && topK < shards {
		scores := rthm.springCandidateScores(addr, senderPos)
		allowed := make([]int, 0, shards)
		for sid := 0; sid < shards; sid++ {
			if sid < len(mask) && mask[sid] > 0 {
				allowed = append(allowed, sid)
			}
		}
		sort.SliceStable(allowed, func(i, j int) bool {
			left := allowed[i]
			right := allowed[j]
			if math.Abs(scores[left]-scores[right]) <= 1e-12 {
				return left < right
			}
			return scores[left] > scores[right]
		})
		keep := make(map[int]bool)
		limit := topK
		if len(allowed) < limit {
			limit = len(allowed)
		}
		for idx := 0; idx < limit; idx++ {
			keep[allowed[idx]] = true
		}
		nextMask := make([]int, shards)
		for sid := 0; sid < shards; sid++ {
			if keep[sid] {
				nextMask[sid] = 1
			}
		}
		mask = nextMask
	}

	return springNormalizeActionMask(mask)
}

func (rthm *RelayCommitteeModule) springBestCandidateShard(
	addr utils.Address,
	senderPos []float64,
	mask []int,
) uint64 {
	normalized := springNormalizeActionMask(mask)
	scores := rthm.springCandidateScores(addr, senderPos)
	bestSid := uint64(0)
	bestScore := math.Inf(-1)
	for sid := 0; sid < params.ShardNum; sid++ {
		if sid >= len(normalized) || normalized[sid] == 0 {
			continue
		}
		score := scores[sid]
		if score > bestScore {
			bestScore = score
			bestSid = uint64(sid)
		}
	}
	return bestSid
}

func (rthm *RelayCommitteeModule) springApplyCandidateGuard(
	addr utils.Address,
	senderPos []float64,
	proposed uint64,
) (uint64, bool) {
	mask := rthm.springBuildCandidateActionMask(addr, senderPos)
	if springActionAllowed(mask, proposed) {
		return proposed, false
	}
	return rthm.springBestCandidateShard(addr, senderPos, mask), true
}

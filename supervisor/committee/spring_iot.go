package committee

import (
	"blockEmulator/core"
	"blockEmulator/params"
	"blockEmulator/utils"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"net/netip"
	"os"
	"strconv"
	"strings"
)

type SpringIOTTxFeature struct {
	TxIndex                 uint64
	StateObject             string
	AnchorObject            string
	AnchorObjects           []string
	DeviceProtocolKey       string
	Features                []float64
	CommunicationCostWeight float64
}

func springIOTEnabled() bool {
	return params.SpringIOTMode == 1
}

func springIOTFeatureFromRow(row map[string]string, priorFrequency int) (SpringIOTTxFeature, bool) {
	txIndex, err := strconv.ParseUint(springIOTCleanCell(row["tx_index"]), 10, 64)
	if err != nil {
		return SpringIOTTxFeature{}, false
	}

	anchorKey := springIOTAnchorKey(row)
	anchorObjects := springIOTAnchorKeys(row, anchorKey)
	protocol := springIOTProtocol(row)
	stateObject := springIOTStateObjectKey(row)
	payload := springIOTNumber(row["srcPayloadSize"]) + springIOTNumber(row["dstPayloadSize"])
	packets := springIOTNumber(row["srcNumPackets"]) + springIOTNumber(row["dstNumPackets"])
	distance := springIOTNumber(row["distance"])
	linkQuality := springIOTNumber(row["link_quality"])

	features := []float64{
		springIOTLogNormalize(payload, 1_000_000.0),
		springIOTLogNormalize(packets, 10_000.0),
		springClamp(distance/60.0, 0.0, 1.0),
		springClamp(1.0-linkQuality, 0.0, 1.0),
		springIOTLogNormalize(float64(priorFrequency), 1000.0),
		springIOTProtocolScore(protocol),
	}

	distanceNorm := springClamp(distance/60.0, 0.0, 1.0)
	linkLoss := springClamp(1.0-linkQuality, 0.0, 1.0)
	trafficWeight := math.Max(0.05, features[0])

	return SpringIOTTxFeature{
		TxIndex:                 txIndex,
		StateObject:             stateObject,
		AnchorObject:            anchorKey,
		AnchorObjects:           anchorObjects,
		DeviceProtocolKey:       anchorKey + "|" + protocol,
		Features:                features,
		CommunicationCostWeight: springClamp(distanceNorm*linkLoss*trafficWeight, 0.0, 1.0),
	}, true
}

func springLoadIOTSidecar(path string) (map[uint64]SpringIOTTxFeature, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("SpringIOTSidecarFile is empty")
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}

	features := make(map[uint64]SpringIOTTxFeature)
	deviceProtocolSeen := make(map[string]int)
	rowIndex := uint64(0)

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		row := make(map[string]string, len(header))
		for idx, name := range header {
			if idx < len(record) {
				row[name] = record[idx]
			}
		}
		if strings.TrimSpace(row["tx_index"]) == "" {
			row["tx_index"] = strconv.FormatUint(rowIndex, 10)
		}

		deviceProtocolKey := springIOTAnchorKey(row) + "|" + springIOTProtocol(row)
		priorFrequency := deviceProtocolSeen[deviceProtocolKey]
		feature, ok := springIOTFeatureFromRow(row, priorFrequency)
		if !ok {
			return nil, fmt.Errorf("invalid IoT sidecar row at index %d", rowIndex)
		}

		features[feature.TxIndex] = feature
		deviceProtocolSeen[deviceProtocolKey] = priorFrequency + 1
		rowIndex++
	}

	return features, nil
}

func (rthm *RelayCommitteeModule) springApplyIOTTxIdentity(tx *core.Transaction) bool {
	if tx == nil || rthm.springIOTByTxIndex == nil {
		return false
	}

	feature, ok := rthm.springIOTByTxIndex[tx.Nonce]
	if !ok || feature.StateObject == "" || feature.AnchorObject == "" {
		return false
	}

	// IoT MDP：链上评估直接使用 sidecar 中的状态对象账户和设备锚点账户，
	// 与 Python offline training（离线训练）和 selectedTxs_iot_full.csv 对齐。
	tx.Sender = feature.StateObject
	tx.Recipient = feature.AnchorObject
	return true
}

func (rthm *RelayCommitteeModule) springIOTFeaturesForTx(tx *core.Transaction) []float64 {
	if tx == nil || rthm.springIOTByTxIndex == nil {
		return nil
	}
	feature, ok := rthm.springIOTByTxIndex[tx.Nonce]
	if !ok {
		return nil
	}
	return feature.Features
}

func (rthm *RelayCommitteeModule) springBuildIOTBatchRelatedMap(txlist []*core.Transaction) map[string]map[string]bool {
	related := make(map[string]map[string]bool)
	for _, tx := range txlist {
		if tx == nil || tx.Sender == "" {
			continue
		}
		key := string(tx.Sender)
		if _, ok := related[key]; !ok {
			related[key] = make(map[string]bool)
		}
		for _, anchor := range rthm.springIOTAnchorsForTx(tx) {
			if anchor == "" || anchor == key {
				continue
			}
			related[key][anchor] = true
		}
	}
	return related
}

func (rthm *RelayCommitteeModule) springSeedIOTAnchorShards(
	txlist []*core.Transaction,
	batchPlacement map[string]uint64,
) {
	for _, tx := range txlist {
		if tx == nil {
			continue
		}

		for _, anchor := range rthm.springIOTAnchorsForTx(tx) {
			if anchor == "" {
				continue
			}
			sid, ok := rthm.springAddrShard[anchor]
			if !ok {
				// IoT anchor（锚点）账户是外部参照对象，不由 PPO（近端策略优化）放置。
				// multi-anchor（多锚点）数据里的设备、网关、云端和服务组锚点都先用
				// hash（哈希）固定分片，供 sender_pos（相关对象分片分布）使用。
				sid = uint64(utils.Addr2Shard(anchor))
				rthm.springAddrShard[anchor] = sid
			}
			if batchPlacement != nil {
				batchPlacement[anchor] = sid
			}
		}
	}
}

func (rthm *RelayCommitteeModule) springIOTAnchorsForTx(tx *core.Transaction) []string {
	if tx == nil {
		return nil
	}
	if rthm.springIOTByTxIndex != nil {
		feature, ok := rthm.springIOTByTxIndex[tx.Nonce]
		if ok && len(feature.AnchorObjects) > 0 {
			return feature.AnchorObjects
		}
	}
	if tx.Recipient != "" {
		return []string{string(tx.Recipient)}
	}
	return nil
}

func springIOTStateObjectKey(row map[string]string) string {
	stateAddress := springIOTCleanCell(row["to_address"])
	if stateAddress != "" {
		return stateAddress
	}
	readableKey := springIOTCleanCell(row["state_object_key"])
	if readableKey != "" {
		return readableKey
	}
	return "iot_state:" + springIOTDeviceKey(row) + "|" + springIOTPeerKey(row) + "|" + springIOTProtocol(row)
}

func springIOTAnchorKey(row map[string]string) string {
	deviceAddress := springIOTCleanCell(row["from_address"])
	if deviceAddress != "" {
		return deviceAddress
	}
	return "iot_device:" + springIOTDeviceKey(row)
}

func springIOTAnchorKeys(row map[string]string, primaryAnchor string) []string {
	raw := springIOTCleanCell(row["anchor_addresses"])
	anchors := make([]string, 0)
	seen := make(map[string]bool)

	add := func(anchor string) {
		anchor = springIOTCleanCell(anchor)
		if anchor == "" || seen[anchor] {
			return
		}
		seen[anchor] = true
		anchors = append(anchors, anchor)
	}

	if raw != "" {
		for _, part := range strings.Split(raw, ";") {
			add(part)
		}
	}
	add(primaryAnchor)
	return anchors
}

func springIOTDeviceKey(row map[string]string) string {
	label := springIOTCleanCell(row["device_label"])
	if label == "" {
		label = "unknown_device"
	}
	mac := strings.ToLower(springIOTCleanCell(row["device_mac"]))
	if mac == "" {
		mac = "unknown_mac"
	}
	return strings.ToLower(label) + ":" + mac
}

func springIOTPeerKey(row map[string]string) string {
	peer := springIOTChoosePeer(row)
	protocol := springIOTProtocol(row)
	service := springIOTCleanCell(row["dstPort"])
	if service == "" {
		service = "any"
	}
	return "iot_peer:" + peer + "|service:" + service + "|proto:" + protocol
}

func springIOTChoosePeer(row map[string]string) string {
	srcIP := springIOTCleanCell(row["srcIp"])
	dstIP := springIOTCleanCell(row["dstIp"])
	if springIOTIsPrivateIP(srcIP) && !springIOTIsPrivateIP(dstIP) {
		return springIOTNormalizeEndpoint(dstIP)
	}
	if springIOTIsPrivateIP(dstIP) && !springIOTIsPrivateIP(srcIP) {
		return springIOTNormalizeEndpoint(srcIP)
	}
	if dstIP != "" {
		return springIOTNormalizeEndpoint(dstIP)
	}
	if srcIP != "" {
		return springIOTNormalizeEndpoint(srcIP)
	}
	peer := springIOTCleanCell(row["to_address"])
	if peer == "" {
		return "unknown_peer"
	}
	return peer
}

func springIOTNormalizeEndpoint(value string) string {
	text := strings.ToLower(springIOTCleanCell(value))
	if text == "" {
		return "unknown_peer"
	}
	return text
}

func springIOTProtocol(row map[string]string) string {
	protocol := strings.ToLower(springIOTCleanCell(row["protocol"]))
	if protocol == "" {
		return "none"
	}
	return protocol
}

func springIOTProtocolScore(protocol string) float64 {
	switch strings.ToLower(protocol) {
	case "tls", "https", "ssl":
		return 1.0
	case "http", "rtsp", "rtmp", "xmpp":
		return 0.8
	case "tcp", "udp", "none":
		return 0.5
	case "dns", "ntp", "stun", "syslog":
		return 0.3
	default:
		return 0.6
	}
}

func springIOTLogNormalize(value float64, scale float64) float64 {
	return springClamp(math.Log1p(math.Max(0.0, value))/math.Log1p(math.Max(1.0, scale)), 0.0, 1.0)
}

func springIOTNumber(value string) float64 {
	v, err := strconv.ParseFloat(springIOTCleanCell(value), 64)
	if err != nil {
		return 0.0
	}
	return v
}

func springIOTCleanCell(value string) string {
	return strings.Trim(strings.TrimSpace(value), "\"")
}

func springIOTIsPrivateIP(value string) bool {
	addr, err := netip.ParseAddr(springIOTCleanCell(value))
	if err != nil {
		return false
	}
	return addr.IsPrivate()
}

func springClamp(value float64, lo float64, hi float64) float64 {
	if value < lo {
		return lo
	}
	if value > hi {
		return hi
	}
	return value
}

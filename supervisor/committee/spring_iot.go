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
	duration := springIOTNumber(row["flowDuration"])
	anchorWeights := springIOTNormalizedWeights(row)
	distances := springIOTNumericList(row, "anchor_distances", "distance", len(anchorWeights))
	linkQualities := springIOTNumericList(row, "anchor_link_qualities", "link_quality", len(anchorWeights))
	normalizedDistances := make([]float64, 0, len(distances))
	for _, distance := range distances {
		normalizedDistances = append(normalizedDistances, springClamp(distance/60.0, 0.0, 1.0))
	}
	normalizedLinkQualities := make([]float64, 0, len(linkQualities))
	for _, linkQuality := range linkQualities {
		normalizedLinkQualities = append(normalizedLinkQualities, springClamp(linkQuality, 0.0, 1.0))
	}
	distanceStats := springIOTWeightedStats(normalizedDistances, anchorWeights)
	linkQualityStats := springIOTWeightedStats(normalizedLinkQualities, anchorWeights)
	trafficRateFeature := springIOTTrafficRateFeature(payload, duration)
	anchorCount := springIOTNumber(row["anchor_count"])
	if anchorCount <= 0 {
		anchorCount = float64(len(anchorWeights))
	}
	if anchorCount <= 0 {
		anchorCount = 1
	}

	features := []float64{
		springClamp(anchorCount/4.0, 0.0, 1.0),
		distanceStats[0],
		distanceStats[1],
		linkQualityStats[0],
		trafficRateFeature,
	}
	anchorShares := springIOTLiteAnchorTypeShares(row, anchorWeights)
	features = append(features, anchorShares["device"])
	features = append(features, anchorShares["edge"])
	features = append(features, anchorShares["cloud"])
	features = append(features, anchorShares["service"])
	features = append(features, springIOTControlProtocolFlag(protocol, row))

	distanceNorm := distanceStats[0]
	linkLoss := springClamp(1.0-linkQualityStats[0], 0.0, 1.0)
	trafficWeight := math.Max(0.05, springIOTLogNormalize(payload+packets+duration, 1_000_000.0))

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

func springIOTTrafficRateFeature(payload float64, duration float64) float64 {
	if duration <= 1e-8 {
		return springIOTLogNormalize(payload, 1_000_000.0)
	}
	return springIOTLogNormalize(payload/duration, 10_000.0)
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

func (rthm *RelayCommitteeModule) springIOTCommunicationCostForTxGroups(txGroups ...[]*core.Transaction) (float64, int) {
	if rthm == nil || rthm.springIOTByTxIndex == nil {
		return 0.0, 0
	}
	seen := make(map[uint64]bool)
	sum := 0.0
	count := 0
	for _, txs := range txGroups {
		for _, tx := range txs {
			if tx == nil || seen[tx.Nonce] {
				continue
			}
			seen[tx.Nonce] = true
			feature, ok := rthm.springIOTByTxIndex[tx.Nonce]
			if !ok {
				continue
			}
			sum += springClamp(feature.CommunicationCostWeight, 0.0, 1.0)
			count++
		}
	}
	return sum, count
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

var springIOTAnchorTypeOrder = []string{
	"device",
	"gateway",
	"cloud_endpoint",
	"local_endpoint",
	"private_endpoint",
	"service_group",
}

var springIOTFlowDirectionOrder = []string{
	"outbound",
	"inbound",
	"owner_not_in_ip_endpoints",
}

var springIOTProtocolGroupOrder = []string{
	"web",
	"secure",
	"infra",
	"control",
	"other",
}

var springIOTRelationGroupOrder = []string{
	"cloud",
	"gateway",
	"local_private",
	"device_multi",
	"service_owner_other",
}

func springIOTNormalizedWeights(row map[string]string) []float64 {
	rawWeights := springIOTSplitList(row["anchor_weights"])
	weights := make([]float64, 0, len(rawWeights))
	for _, raw := range rawWeights {
		weight := springIOTNumber(raw)
		if weight < 0 {
			weight = 0
		}
		weights = append(weights, weight)
	}

	anchorCount := len(springIOTSplitList(row["anchor_addresses"]))
	if anchorCount == 0 {
		anchorCount = len(springIOTSplitList(row["anchor_types"]))
	}
	if len(weights) == 0 {
		if anchorCount <= 0 {
			anchorCount = 1
		}
		for i := 0; i < anchorCount; i++ {
			weights = append(weights, 1.0)
		}
	}

	total := 0.0
	for _, weight := range weights {
		total += weight
	}
	if total <= 0 {
		uniform := 1.0 / float64(len(weights))
		for i := range weights {
			weights[i] = uniform
		}
		return weights
	}
	for i := range weights {
		weights[i] = weights[i] / total
	}
	return weights
}

func springIOTSplitList(value string) []string {
	text := springIOTCleanCell(value)
	if text == "" {
		return nil
	}
	parts := strings.Split(text, ";")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = springIOTCleanCell(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func springIOTNumericList(row map[string]string, field string, fallbackField string, expectedLen int) []float64 {
	rawValues := springIOTSplitList(row[field])
	values := make([]float64, 0, len(rawValues))
	for _, raw := range rawValues {
		values = append(values, springIOTNumber(raw))
	}
	if len(values) == 0 {
		values = append(values, springIOTNumber(row[fallbackField]))
	}
	if expectedLen <= 0 {
		expectedLen = 1
	}
	for len(values) < expectedLen {
		last := 0.0
		if len(values) > 0 {
			last = values[len(values)-1]
		}
		values = append(values, last)
	}
	if len(values) > expectedLen {
		values = values[:expectedLen]
	}
	return values
}

func springIOTWeightedStats(values []float64, weights []float64) [3]float64 {
	if len(values) == 0 {
		return [3]float64{0, 0, 0}
	}
	if len(weights) != len(values) || len(weights) == 0 {
		weights = make([]float64, len(values))
		for i := range weights {
			weights[i] = 1.0 / float64(len(values))
		}
	}

	total := 0.0
	for _, weight := range weights {
		if weight > 0 {
			total += weight
		}
	}
	if total <= 0 {
		total = 1.0
		for i := range weights {
			weights[i] = 1.0 / float64(len(values))
		}
	}

	avg := 0.0
	minValue := values[0]
	maxValue := values[0]
	for idx, value := range values {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
		weight := weights[idx]
		if weight < 0 {
			weight = 0
		}
		avg += value * weight
	}
	avg = avg / total
	return [3]float64{
		springClamp(avg, 0.0, 1.0),
		springClamp(minValue, 0.0, 1.0),
		springClamp(maxValue, 0.0, 1.0),
	}
}

func springIOTAnchorTypeDistribution(row map[string]string, weights []float64) []float64 {
	rawTypes := springIOTSplitList(row["anchor_types"])
	if len(rawTypes) == 0 {
		rawType := springIOTCleanCell(row["peer_anchor_type"])
		if rawType != "" {
			rawTypes = []string{rawType}
		}
	}
	if len(rawTypes) == 0 {
		rawTypes = []string{"device"}
	}
	for len(weights) < len(rawTypes) {
		weights = append(weights, 1.0)
	}
	if len(weights) > len(rawTypes) {
		weights = weights[:len(rawTypes)]
	}

	total := 0.0
	for _, weight := range weights {
		if weight > 0 {
			total += weight
		}
	}
	if total <= 0 {
		total = 1.0
		for i := range weights {
			weights[i] = 1.0 / float64(len(rawTypes))
		}
	}

	dist := make([]float64, len(springIOTAnchorTypeOrder))
	index := make(map[string]int, len(springIOTAnchorTypeOrder))
	for idx, name := range springIOTAnchorTypeOrder {
		index[name] = idx
	}
	for idx, rawType := range rawTypes {
		group := springIOTAnchorTypeGroup(rawType)
		if pos, ok := index[group]; ok {
			weight := weights[idx]
			if weight < 0 {
				weight = 0
			}
			dist[pos] += weight / total
		}
	}
	for idx := range dist {
		dist[idx] = springClamp(dist[idx], 0.0, 1.0)
	}
	return dist
}

func springIOTLiteAnchorTypeShares(row map[string]string, weights []float64) map[string]float64 {
	dist := springIOTAnchorTypeDistribution(row, weights)
	index := make(map[string]int, len(springIOTAnchorTypeOrder))
	for idx, name := range springIOTAnchorTypeOrder {
		index[name] = idx
	}
	edgeShare := dist[index["gateway"]] +
		dist[index["local_endpoint"]] +
		dist[index["private_endpoint"]]
	return map[string]float64{
		"device":  springClamp(dist[index["device"]], 0.0, 1.0),
		"edge":    springClamp(edgeShare, 0.0, 1.0),
		"cloud":   springClamp(dist[index["cloud_endpoint"]], 0.0, 1.0),
		"service": springClamp(dist[index["service_group"]], 0.0, 1.0),
	}
}

func springIOTAnchorTypeGroup(value string) string {
	text := strings.ToLower(springIOTCleanCell(value))
	if strings.Contains(text, "gateway") {
		return "gateway"
	}
	if strings.Contains(text, "cloud") {
		return "cloud_endpoint"
	}
	if strings.Contains(text, "local") {
		return "local_endpoint"
	}
	if strings.Contains(text, "private") {
		return "private_endpoint"
	}
	if strings.Contains(text, "service") {
		return "service_group"
	}
	return "device"
}

func springIOTControlProtocolFlag(protocol string, row map[string]string) float64 {
	protocol = strings.ToLower(protocol)
	if springIOTBool(row["is_control_protocol"]) > 0 {
		return 1.0
	}
	switch protocol {
	case "icmp", "arp", "dhcp", "mdns", "ssdp", "igmp":
		return 1.0
	default:
		return 0.0
	}
}

func springIOTFlowDirectionGroup(row map[string]string) string {
	direction := strings.ToLower(springIOTCleanCell(row["flow_direction"]))
	for _, item := range springIOTFlowDirectionOrder {
		if direction == item {
			return direction
		}
	}
	if strings.HasPrefix(direction, "in") {
		return "inbound"
	}
	if strings.HasPrefix(direction, "out") {
		return "outbound"
	}
	return "owner_not_in_ip_endpoints"
}

func springIOTProtocolGroup(protocol string, row map[string]string) string {
	protocol = strings.ToLower(protocol)
	if springIOTBool(row["is_control_protocol"]) > 0 {
		return "control"
	}
	switch protocol {
	case "tls", "https", "ssl":
		return "secure"
	case "http", "rtsp", "rtmp", "xmpp", "mqtt", "coap":
		return "web"
	case "dns", "ntp", "stun", "syslog":
		return "infra"
	case "icmp", "arp", "dhcp", "mdns", "ssdp", "igmp":
		return "control"
	default:
		return "other"
	}
}

func springIOTRelationGroup(row map[string]string) string {
	relation := strings.ToLower(springIOTCleanCell(row["relation_type"]))
	if strings.Contains(relation, "cloud") {
		return "cloud"
	}
	if strings.Contains(relation, "gateway") {
		return "gateway"
	}
	if strings.Contains(relation, "local") || strings.Contains(relation, "private") {
		return "local_private"
	}
	if strings.Contains(relation, "device_to_device") || strings.Contains(relation, "multi_anchor") {
		return "device_multi"
	}
	return "service_owner_other"
}

func springIOTOneHot(order []string, value string) []float64 {
	out := make([]float64, 0, len(order))
	for _, item := range order {
		if item == value {
			out = append(out, 1.0)
		} else {
			out = append(out, 0.0)
		}
	}
	return out
}

func springIOTBool(value string) float64 {
	text := strings.ToLower(springIOTCleanCell(value))
	if text == "1" || text == "true" || text == "yes" || text == "y" {
		return 1.0
	}
	return 0.0
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

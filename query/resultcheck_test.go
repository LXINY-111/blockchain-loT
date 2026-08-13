package query

import (
	"blockEmulator/core"
	"blockEmulator/params"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/big"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestFinalResult(t *testing.T) {
	// This is an integration check for a completed BlockEmulator run, not a
	// hermetic unit test. Opt in explicitly so parallel package tests cannot
	// observe another test's transient database directory.
	if os.Getenv("BLOCKEMULATOR_CHECK_FINAL_RESULT") != "1" {
		t.Skip("set BLOCKEMULATOR_CHECK_FINAL_RESULT=1 to verify a completed experiment")
	}

	// Go tests run with query/ as their working directory. Load the project
	// root paramsConfig.json so this check targets the timestamped run that
	// BlockEmulator actually executed instead of the source-code placeholder.
	queryDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("get query working directory: %v", err)
	}
	projectRoot := filepath.Clean(filepath.Join(queryDir, ".."))
	if err := os.Chdir(projectRoot); err != nil {
		t.Fatalf("change to project root: %v", err)
	}
	params.ReadConfigFile()
	if err := os.Chdir(queryDir); err != nil {
		t.Fatalf("restore query working directory: %v", err)
	}

	// check the final result after running BlockEmulator
	firstMptPath := queryTestPath(params.DatabaseWrite_path + "mptDB/ldb/s0/n0")
	if _, err := os.Stat(firstMptPath); err != nil {
		t.Skipf("skip final result check because BlockEmulator database is not present: %s", firstMptPath)
	}

	// get the result from Dataset
	accountBalance := loadFinalResultFromDataset()
	acCorrect := make(map[string]bool)

	// convert keys to list
	accounts := make([]string, len(accountBalance))
	i := 0
	for key := range accountBalance {
		accounts[i] = key
		i++
	}

	// check the result from BlockEmulator
	acFound := make(map[string]int)
	for sid := 0; sid < params.ShardNum; sid++ {
		mptfp := queryTestPath(params.DatabaseWrite_path + "mptDB/ldb/s" + strconv.FormatUint(uint64(sid), 10) + "/n0")
		chaindbfp := queryTestPath(params.DatabaseWrite_path + fmt.Sprintf("chainDB/S%d_N%d", sid, 0))
		newest := QueryNewestBlock(uint64(sid), 0)
		fmt.Printf(
			"Shard %d newest block=%d stateRoot=%x\n",
			sid,
			newest.Header.Number,
			newest.Header.StateRoot,
		)
		aslist := QueryAccountStateList(chaindbfp, mptfp, uint64(sid), 0, accounts)
		for idx, as := range aslist {
			if as != nil {
				acFound[accounts[idx]]++
			}
			if as != nil && as.Balance.Cmp(accountBalance[accounts[idx]]) == 0 {
				acCorrect[accounts[idx]] = true
			}
		}
	}
	duplicateAccounts := 0
	for _, count := range acFound {
		if count > 1 {
			duplicateAccounts++
		}
	}
	fmt.Println("Results from BlockEmulator: # of correct accounts", len(acCorrect))
	fmt.Println("Results from BlockEmulator: # of found accounts", len(acFound))
	fmt.Println("Results from BlockEmulator: # of duplicate accounts", duplicateAccounts)
	if len(acCorrect) == len(accountBalance) {
		fmt.Println("test pass")
	} else if len(accountBalance)-len(acCorrect) < params.BrokerNum {
		fmt.Printf("%d err accounts, they maybe brokers", len(accountBalance)-len(acCorrect))
	} else {
		log.Panic("Err, too many wrong accounts", len(accountBalance)-len(acCorrect))
	}
}

func queryTestPath(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return "../" + path
}

func data2tx(data []string, nonce uint64) (*core.Transaction, bool) {
	if data[6] == "0" && data[7] == "0" && len(data[3]) > 16 && len(data[4]) > 16 && data[3] != data[4] {
		val, ok := new(big.Int).SetString(data[8], 10)
		if !ok {
			log.Panic("new int failed\n")
		}
		tx := core.NewTransaction(data[3][2:], data[4][2:], val, nonce, time.Now())
		return tx, true
	}
	return &core.Transaction{}, false
}

func loadFinalResultFromDataset() map[string]*big.Int {
	accountBalance := make(map[string]*big.Int)
	txfile, err := os.Open("../" + params.DatasetFile)
	if err != nil {
		log.Panic(err)
	}
	defer txfile.Close()
	reader := csv.NewReader(txfile)

	var sidecarFile *os.File
	var sidecarReader *csv.Reader
	var sidecarColumns map[string]int
	if params.SpringIOTMode == 1 || params.SpringIOTIdentityMode == 1 {
		sidecarFile, err = os.Open(queryTestPath(params.SpringIOTSidecarFile))
		if err != nil {
			log.Panic(err)
		}
		defer sidecarFile.Close()
		sidecarReader = csv.NewReader(sidecarFile)
		header, err := sidecarReader.Read()
		if err != nil {
			log.Panic(err)
		}
		sidecarColumns = make(map[string]int, len(header))
		for idx, name := range header {
			sidecarColumns[strings.TrimSpace(name)] = idx
		}
		for _, name := range []string{"tx_index", "from_address", "to_address"} {
			if _, ok := sidecarColumns[name]; !ok {
				log.Panicf("IoT sidecar is missing column %q", name)
			}
		}
	}

	nowDataNum := 0
	sourceValidNum := 0
	for {
		data, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Panic(err)
		}
		if tx, ok := data2tx(data, uint64(nowDataNum)); ok {
			if sourceValidNum >= params.DatasetStartTx && nowDataNum == params.TotalDataSize {
				break
			}

			if sidecarReader != nil {
				identityRow, err := sidecarReader.Read()
				if err != nil {
					log.Panic(err)
				}
				txIndex, err := strconv.Atoi(strings.TrimSpace(identityRow[sidecarColumns["tx_index"]]))
				if err != nil || txIndex != sourceValidNum {
					log.Panicf(
						"IoT sidecar index mismatch: row=%q parsed=%d expected=%d err=%v",
						identityRow[sidecarColumns["tx_index"]],
						txIndex,
						sourceValidNum,
						err,
					)
				}
				tx.Sender = strings.TrimSpace(identityRow[sidecarColumns["to_address"]])
				tx.Recipient = strings.TrimSpace(identityRow[sidecarColumns["from_address"]])
				if tx.Sender == "" || tx.Recipient == "" {
					log.Panicf("IoT sidecar identity is empty at tx_index=%d", txIndex)
				}
			}

			if sourceValidNum < params.DatasetStartTx {
				sourceValidNum++
				continue
			}
			sourceValidNum++
			nowDataNum++
			if _, ok := accountBalance[tx.Sender]; !ok {
				accountBalance[tx.Sender] = new(big.Int)
				accountBalance[tx.Sender].Add(accountBalance[tx.Sender], params.Init_Balance)
			}

			if _, ok := accountBalance[tx.Recipient]; !ok {
				accountBalance[tx.Recipient] = new(big.Int)
				accountBalance[tx.Recipient].Add(accountBalance[tx.Recipient], params.Init_Balance)
			}
			if accountBalance[tx.Sender].Cmp(tx.Value) != -1 {
				accountBalance[tx.Sender].Sub(accountBalance[tx.Sender], tx.Value)
				accountBalance[tx.Recipient].Add(accountBalance[tx.Recipient], tx.Value)
			}
		}
	}
	fmt.Println("Results from dataset file: # of accounts", len(accountBalance))

	return accountBalance
}

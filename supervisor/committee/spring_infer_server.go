package committee

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

type springInferServerRequest struct {
	RequestID          uint64                 `json:"request_id"`
	Shards             int                    `json:"shards"`
	IOTFeatureDim      int                    `json:"iot_feature_dim,omitempty"`
	AllowModelInit     bool                   `json:"allow_model_init"`
	Sample             bool                   `json:"sample"`
	Model              string                 `json:"model"`
	ExpectedCheckpoint map[string]interface{} `json:"expected_checkpoint,omitempty"`
	Items              []SpringBatchInferItem `json:"items"`
}

type springInferServerResponse struct {
	RequestID uint64                   `json:"request_id"`
	Items     []SpringBatchInferResult `json:"items"`
	Error     string                   `json:"error,omitempty"`
}

var springInferServerMu sync.Mutex
var springInferServerCmd *exec.Cmd
var springInferServerIn io.WriteCloser
var springInferServerOut *bufio.Reader
var springInferServerErrFile *os.File

func springDefaultPythonCmd() string {
	pythonCmd := os.Getenv("SPRING_PYTHON")
	if pythonCmd != "" {
		return pythonCmd
	}
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

// 运行目录只保存配置和结果；可显式引用项目中唯一一份 Python 源码。
// 未设置时保持旧目录布局，旧数据的启动方式不受影响。
func springPythonScript(name string) string {
	if directory := strings.TrimSpace(os.Getenv("SPRING_PYTHON_DIR")); directory != "" {
		return filepath.Join(directory, name)
	}
	return filepath.Join("spring_lite", name)
}

func springStopInferServerLocked() {
	if springInferServerIn != nil {
		_ = springInferServerIn.Close()
		springInferServerIn = nil
	}

	if springInferServerCmd != nil && springInferServerCmd.Process != nil {
		_ = springInferServerCmd.Process.Kill()
		_, _ = springInferServerCmd.Process.Wait()
	}
	springInferServerCmd = nil
	springInferServerOut = nil

	if springInferServerErrFile != nil {
		_ = springInferServerErrFile.Close()
		springInferServerErrFile = nil
	}
}

func springEnsureInferServerLocked() error {
	if springInferServerCmd != nil && springInferServerIn != nil && springInferServerOut != nil {
		return nil
	}

	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		return err
	}

	pythonCmd := springDefaultPythonCmd()

	cmd := exec.Command(
		pythonCmd,
		"-u",
		springPythonScript("infer_server.py"),
	)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	errPath := filepath.Join("spring_io", "infer_server_stderr.log")
	errFile, err := os.OpenFile(errPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	cmd.Stderr = errFile

	if err := cmd.Start(); err != nil {
		_ = errFile.Close()
		return err
	}

	springInferServerCmd = cmd
	springInferServerIn = stdin
	springInferServerOut = bufio.NewReader(stdout)
	springInferServerErrFile = errFile

	return nil
}

func springCallInferServer(req springInferServerRequest) (springInferServerResponse, error) {
	springInferServerMu.Lock()
	defer springInferServerMu.Unlock()

	if err := springEnsureInferServerLocked(); err != nil {
		return springInferServerResponse{}, err
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return springInferServerResponse{}, err
	}

	reqBytes = append(reqBytes, '\n')

	if _, err := springInferServerIn.Write(reqBytes); err != nil {
		springStopInferServerLocked()
		return springInferServerResponse{}, err
	}

	line, err := springInferServerOut.ReadBytes('\n')
	if err != nil {
		springStopInferServerLocked()
		return springInferServerResponse{}, err
	}

	var resp springInferServerResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		return springInferServerResponse{}, fmt.Errorf("bad infer_server response: %v, raw=%s", err, string(line))
	}

	if resp.Error != "" {
		return resp, fmt.Errorf(resp.Error)
	}

	return resp, nil
}

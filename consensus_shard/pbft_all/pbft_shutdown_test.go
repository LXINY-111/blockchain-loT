package pbft_all

import (
	"sync"
	"testing"
	"time"
)

func newShutdownTestNode() *PbftConsensusNode {
	node := &PbftConsensusNode{
		stopCh:    make(chan struct{}),
		stoppedCh: make(chan struct{}),
	}
	node.conditionalVarpbftLock = *sync.NewCond(&node.pbftLock)
	return node
}

func TestShutdownCompletionWaitsForAdmittedHandlers(t *testing.T) {
	node := newShutdownTestNode()
	handlerStarted := make(chan struct{})
	releaseHandler := make(chan struct{})
	if !node.launchTrackedHandler(func() {
		close(handlerStarted)
		<-releaseHandler
	}) {
		t.Fatal("expected handler admission before shutdown")
	}
	<-handlerStarted
	if !node.beginStop() {
		t.Fatal("expected shutdown to start")
	}

	go func() {
		node.inflightHandlers.Wait()
		node.publishShutdownComplete()
	}()
	waitReturned := make(chan struct{})
	go func() {
		node.waitForShutdownComplete()
		close(waitReturned)
	}()
	select {
	case <-waitReturned:
		t.Fatal("shutdown completion returned before the handler drained")
	case <-time.After(50 * time.Millisecond):
	}
	close(releaseHandler)
	select {
	case <-waitReturned:
	case <-time.After(time.Second):
		t.Fatal("shutdown completion did not publish after handler drain")
	}
}

func TestShutdownWaitsForAdmittedHandlersAndRejectsNewWork(t *testing.T) {
	node := newShutdownTestNode()
	handlerStarted := make(chan struct{})
	releaseHandler := make(chan struct{})

	if !node.launchTrackedHandler(func() {
		close(handlerStarted)
		<-releaseHandler
	}) {
		t.Fatal("expected handler admission before shutdown")
	}
	<-handlerStarted

	if !node.beginStop() {
		t.Fatal("expected the first shutdown request to begin shutdown")
	}
	select {
	case <-node.stopCh:
	default:
		t.Fatal("shutdown channel was not closed")
	}
	if node.launchTrackedHandler(func() {}) {
		t.Fatal("handler was admitted after shutdown began")
	}
	if node.beginStop() {
		t.Fatal("duplicate shutdown request should be ignored")
	}

	drained := make(chan struct{})
	go func() {
		node.inflightHandlers.Wait()
		close(drained)
	}()

	select {
	case <-drained:
		t.Fatal("shutdown drain returned before the admitted handler finished")
	default:
	}

	close(releaseHandler)
	select {
	case <-drained:
	case <-time.After(time.Second):
		t.Fatal("shutdown drain did not return after the handler finished")
	}
}

func TestBeginStopWakesPBFTStageWaiters(t *testing.T) {
	node := newShutdownTestNode()
	waiterReady := make(chan struct{})
	waiterDone := make(chan struct{})

	node.pbftLock.Lock()
	go func() {
		node.pbftLock.Lock()
		close(waiterReady)
		for !node.stopSignal.Load() {
			node.conditionalVarpbftLock.Wait()
		}
		node.pbftLock.Unlock()
		close(waiterDone)
	}()
	node.pbftLock.Unlock()
	<-waiterReady

	if !node.beginStop() {
		t.Fatal("expected shutdown to start")
	}
	select {
	case <-waiterDone:
	case <-time.After(time.Second):
		t.Fatal("PBFT stage waiter was not woken by shutdown")
	}
}

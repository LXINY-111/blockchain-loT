// storage is a key-value database and its interfaces indeed
// the information of block will be saved in storage

package storage

import (
	"blockEmulator/core"
	"blockEmulator/params"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/boltdb/bolt"
)

type Storage struct {
	dbFilePath            string // path to the database
	blockBucket           string // bucket in bolt database
	blockHeaderBucket     string // bucket in bolt database
	newestBlockHashBucket string // bucket in bolt database
	executedTxBucket      string // committed transaction hashes, used to reject replay
	DataBase              *bolt.DB
}

// new a storage, build a bolt datase
func NewStorage(dbFp string, cc *params.ChainConfig) *Storage {
	dir := params.DatabaseWrite_path + "chainDB"
	errMkdir := os.MkdirAll(dir, os.ModePerm)
	if errMkdir != nil {
		log.Panic(errMkdir)
	}

	s := &Storage{
		dbFilePath:            dbFp,
		blockBucket:           "block",
		blockHeaderBucket:     "blockHeader",
		newestBlockHashBucket: "newestBlockHash",
		executedTxBucket:      "executedTransaction",
	}

	db, err := bolt.Open(s.dbFilePath, 0600, nil)
	if err != nil {
		log.Panic(err)
	}

	// create buckets
	err = db.Update(func(tx *bolt.Tx) error {
		blockBucket, err := tx.CreateBucketIfNotExists([]byte(s.blockBucket))
		if err != nil {
			return fmt.Errorf("create blocksBucket: %w", err)
		}

		_, err = tx.CreateBucketIfNotExists([]byte(s.blockHeaderBucket))
		if err != nil {
			return fmt.Errorf("create blockHeaderBucket: %w", err)
		}

		_, err = tx.CreateBucketIfNotExists([]byte(s.newestBlockHashBucket))
		if err != nil {
			return fmt.Errorf("create newestBlockHashBucket: %w", err)
		}

		executedBucket, err := tx.CreateBucketIfNotExists([]byte(s.executedTxBucket))
		if err != nil {
			return fmt.Errorf("create executedTransaction bucket: %w", err)
		}
		// Backfill the replay index for databases created by older versions.
		// The operation is idempotent and keeps existing experiment databases readable.
		if err := blockBucket.ForEach(func(_, encoded []byte) error {
			if encoded == nil {
				return nil
			}
			block := core.DecodeB(encoded)
			for _, transaction := range block.Body {
				if transaction != nil && len(transaction.TxHash) > 0 {
					if err := executedBucket.Put(transaction.TxHash, []byte{1}); err != nil {
						return err
					}
				}
			}
			return nil
		}); err != nil {
			return fmt.Errorf("backfill executed transactions: %w", err)
		}

		return nil
	})
	if err != nil {
		_ = db.Close()
		log.Panic(err)
	}
	s.DataBase = db
	return s
}

// update the newest block in the database
func (s *Storage) UpdateNewestBlock(newestbhash []byte) {
	err := s.DataBase.Update(func(tx *bolt.Tx) error {
		nbhBucket := tx.Bucket([]byte(s.newestBlockHashBucket))
		// the bucket has the only key "OnlyNewestBlock"
		err := nbhBucket.Put([]byte("OnlyNewestBlock"), newestbhash)
		if err != nil {
			log.Panic()
		}
		return nil
	})
	if err != nil {
		log.Panic()
	}
	fmt.Println("The newest block is updated")
}

// add a blockheader into the database
func (s *Storage) AddBlockHeader(blockhash []byte, bh *core.BlockHeader) {
	err := s.DataBase.Update(func(tx *bolt.Tx) error {
		bhbucket := tx.Bucket([]byte(s.blockHeaderBucket))
		err := bhbucket.Put(blockhash, bh.Encode())
		if err != nil {
			log.Panic()
		}
		return nil
	})
	if err != nil {
		log.Panic()
	}
}

// add a block into the database
func (s *Storage) AddBlock(b *core.Block) error {
	if b == nil || b.Header == nil {
		return errors.New("cannot store a nil block or header")
	}
	err := s.DataBase.Update(func(tx *bolt.Tx) error {
		bbucket := tx.Bucket([]byte(s.blockBucket))
		bhbucket := tx.Bucket([]byte(s.blockHeaderBucket))
		nbhBucket := tx.Bucket([]byte(s.newestBlockHashBucket))
		executedBucket := tx.Bucket([]byte(s.executedTxBucket))
		if bbucket == nil || bhbucket == nil || nbhBucket == nil || executedBucket == nil {
			return errors.New("storage bucket is missing")
		}
		if err := bbucket.Put(b.Hash, b.Encode()); err != nil {
			return err
		}
		if err := bhbucket.Put(b.Hash, b.Header.Encode()); err != nil {
			return err
		}
		if err := nbhBucket.Put([]byte("OnlyNewestBlock"), b.Hash); err != nil {
			return err
		}
		for _, transaction := range b.Body {
			if transaction != nil && len(transaction.TxHash) > 0 {
				if err := executedBucket.Put(transaction.TxHash, []byte{1}); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("store block: %w", err)
	}
	fmt.Println("Block is added")
	return nil
}

// HasTransaction reports whether this shard has already committed txHash.
// The index is persisted in BoltDB so replay protection survives node restarts.
func (s *Storage) HasTransaction(txHash []byte) (bool, error) {
	if len(txHash) == 0 {
		return false, errors.New("transaction hash is empty")
	}
	found := false
	err := s.DataBase.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(s.executedTxBucket))
		if bucket == nil {
			return errors.New("executedTransaction bucket is missing")
		}
		found = bucket.Get(txHash) != nil
		return nil
	})
	return found, err
}

// read a blockheader from the database
func (s *Storage) GetBlockHeader(bhash []byte) (*core.BlockHeader, error) {
	var res *core.BlockHeader
	err := s.DataBase.View(func(tx *bolt.Tx) error {
		bhbucket := tx.Bucket([]byte(s.blockHeaderBucket))
		bh_encoded := bhbucket.Get(bhash)
		if bh_encoded == nil {
			return errors.New("the block is not existed")
		}
		res = core.DecodeBH(bh_encoded)
		return nil
	})
	return res, err
}

// read a block from the database
func (s *Storage) GetBlock(bhash []byte) (*core.Block, error) {
	var res *core.Block
	err := s.DataBase.View(func(tx *bolt.Tx) error {
		bbucket := tx.Bucket([]byte(s.blockBucket))
		b_encoded := bbucket.Get(bhash)
		if b_encoded == nil {
			return errors.New("the block is not existed")
		}
		res = core.DecodeB(b_encoded)
		return nil
	})
	return res, err
}

// read the Newest block hash
func (s *Storage) GetNewestBlockHash() ([]byte, error) {
	var nhb []byte
	err := s.DataBase.View(func(tx *bolt.Tx) error {
		bhbucket := tx.Bucket([]byte(s.newestBlockHashBucket))
		// the bucket has the only key "OnlyNewestBlock"
		nhb = bhbucket.Get([]byte("OnlyNewestBlock"))
		if nhb == nil {
			return errors.New("cannot find the newest block hash")
		}
		return nil
	})
	return nhb, err
}

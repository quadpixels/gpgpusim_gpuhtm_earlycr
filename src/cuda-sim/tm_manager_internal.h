/***************************************** 
 
  tm_manager_internal.h
 
  encapsulates data required by transactional memory
  includes restore point, memory values, functionality
  for manipulating g_thread and tm state
 
 
****************************************/


#ifndef __TM_MANAGER_INTERNAL_H
#define __TM_MANAGER_INTERNAL_H

#include <vector>

#include "tm_manager.h"
#include "../abstract_hardware_model.h" 
#include "../option_parser.h"
#include "ptx_sim.h"
#include "memory.h"
#include "bloomfilter.h"
#include "../gpgpu-sim/histogram.h"

#include <list>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <algorithm> 
#include <math.h>

#define TM_MEM_BUCKET_SIZE 6 // log2(number of bytes per bucket) 
#define CACHE_LINE_SIZE 7 // log2 (cache line size ) used to collect statistics on cache lines accessed by transactions
typedef std::set<addr_t> addr_set_t;
// typedef std::set<unsigned> tuid_set_t;
typedef std::unordered_set<unsigned> tuid_set_t;
typedef tr1_hash_map<addr_t, int> addr_version_set_t;

class CartographerTimeSeries;

class tm_manager : public tm_manager_inf
{
	friend class shader_core_ctx;
	friend class Cartographer;
	friend class CartographerTimeSeries;
	friend class tx_log_walker;
	friend class tx_log_walker_warpc;
	friend class tm_warp_info;
	friend class ptx_thread_info;
public:
	tm_manager(	ptx_thread_info *thread, bool timing_mode );
   virtual ~tm_manager();

	virtual void start();
	virtual bool tm_access( memory_space* mem, memory_space_t space, bool rd, addr_t addr, void* vp, int nbytes, tm_access_uarch_info& uarch_info ); // return false if validation failed
	virtual void abort();
	virtual bool commit( bool auto_self_abort ); // return false if commit failed 
   virtual void accessmode( int readmode, int writemode ); 
   virtual void add_rollback_insn( unsigned insn_count ); // track the number of rolled back instruction at abort 
   virtual void add_committed_insn( unsigned insn_count ); // track the number of committed instructions at commit

   virtual bool get_read_conflict_detection() const { return m_read_conflict_detection; }
   virtual bool get_write_conflict_detection() const { return m_write_conflict_detection; }
   virtual bool get_version_management() const { return m_version_management; }

   virtual unsigned get_n_read() const { return m_n_read; }
   virtual unsigned get_n_write() const { return m_n_write; }

   // interface for timing model validation and commit -- only for value-based tm
   virtual bool validate_addr( addr_t addr ); // validate a single word 
   virtual void commit_addr( addr_t addr ); // commit a single word 
   virtual void commit_core_side( ); // commit a transaction on the core side 
   virtual void validate_or_crash( ); // validate a transaction and crash if it is not valid

   // detect conflict between this transaction and the other 
   virtual bool has_conflict_with( tm_manager_inf * other_tx ) {
      assert(0); // invalid for baseline transaction manager
   }

   // validate entire read-set (return true if pass)
   virtual bool validate_all( bool useTemporalCD ) {
      assert(0); // invalid for baseline transaction manager
   }

   // share the global memory view with the given transaction 
   virtual void share_gmem_view( tm_manager_inf* other_tx ) {
      assert(m_gmem_view_tx == NULL); // only do the assignment once 
      m_gmem_view_tx = dynamic_cast<tm_manager*>(other_tx); 
   }

	void print_tm_mem(FILE *fp);
	void print_read_write_set(FILE *fp); 

   size_t get_access_size() const { return (m_read_word_set.size() + m_write_word_set.size()); }
   bool get_read_conflict_detection() { return m_read_conflict_detection; }
   bool get_write_conflict_detection() { return m_write_conflict_detection; }

protected:
   // tm policy specific code
   virtual void at_start(); 
   virtual bool at_access( memory_space *mem, bool potential_conflicting, bool rd, addr_t addr, int nbytes ); // return true if tx needs to self-abort 
   virtual void at_abort(); 
   virtual bool at_commit_validation(); // detect + resolve any conflicts: return true if tx needs to self-abort  
   virtual void at_commit_success(); // after conflicts are resolved 

	void write_stats();
public:
	class trans_stats{
	public:
		typedef std::set<unsigned> tm_stats_set;
		tm_stats_set m_cache_lines_accessed;
		//tm_stats_map  m_total_read_bytes;
		//tm_stats_map  m_total_written_bytes;
		int m_abort; // 1 = transaction was aborted
		int m_overflow;
		trans_stats() 
      : m_abort(0), m_overflow(0)
		{ }
	};

public:
	class access_record {
	public:
		access_record( const access_record &another );
		access_record( tm_manager *parent, memory_space *mem, bool rd, addr_t addr, void *data, unsigned nbytes );
		~access_record();
		addr_t getaddr(){return m_addr;}
		unsigned getsize (){return m_nbytes;}
		void* getvalue(){return m_bytes;}
      bool contain_addr( addr_t addr, unsigned bytes ); 
      memory_space* get_memory_space() const { return m_mem; } 
		void commit();
		void commit_word( addr_t addr, unsigned bytes );
      void print(FILE *fout);
		
	public:
		tm_manager *m_parent;
		bool 		m_rd; // true = read, false = write
		addr_t 		m_addr;
		unsigned 	m_nbytes;
		void *		m_bytes;
		memory_space *m_mem;
	};
protected:
	std::list<access_record> m_write_data;
	tuid_set_t m_conflict_tuids;
	trans_stats m_stats;

   void committer_win_conflict_resolution(); 

	struct tm_mem_bucket 
	{
		tm_mem_bucket() 
		{
			m_modified = 0;
			assert( (1<<TM_MEM_BUCKET_SIZE) == 8*sizeof(unsigned long long) );
			m_data.resize( pow(2,TM_MEM_BUCKET_SIZE) );
		}
		unsigned long long m_modified;
		std::vector<unsigned char> m_data;
	};
	typedef std::map<unsigned,tm_mem_bucket> tm_mem_hash_t;
	typedef std::map<memory_space*, tm_mem_hash_t> tm_mem_t;
	tm_mem_t m_tm_mem;
   tm_manager * m_gmem_view_tx; 

   // update read/write sets in different granularities 
   void update_access_sets(addr_t addr, unsigned nbytes, bool rd, bool detect_conflict); 

   // read/write set in byte granularity 
	addr_set_t m_read_set;
	addr_set_t m_write_set; // only record the potentially conflicting ones

   // read/write set in word (32-bit) granularity 
   addr_set_t m_read_word_set; 
   addr_set_t m_write_word_set;  // only record the potentially conflicting ones
   addr_set_t m_access_word_set; // both read and write 

   // read/write set in block (configurable #bytes) granularity 
   addr_set_t m_read_block_set; 
   addr_set_t m_write_block_set; 
   addr_set_t m_access_block_set; // both read and write 

   // complete write set in word (32-bit) granularity - including non-conflicting writes 
   addr_set_t m_buffered_write_word_set; 

   // warp-level write_word_set to detect intra warp global memory communication 
   addr_set_t m_warp_level_write_word_set;  // only record the potentially conflicting ones

   // read access to transactional data 
   addr_set_t m_raw_set; // footprint 
   unsigned m_raw_access; // bandwidth 

   // bandwidth data 
   unsigned m_n_read;  // count reads the require conflict detection
   unsigned m_n_read_all;
   unsigned m_n_write; // count writes that require buffering 
   unsigned m_n_rewrite; 

   // access mode 
   bool m_read_conflict_detection; 
   bool m_write_conflict_detection;
   bool m_version_management; 

   unsigned long long m_start_cycle; // when the transaction called txbegin()
   unsigned long long m_first_read_cycle; // when the transaction first load from memory 

   // full access log - all conflictable read and all buffered writes
public:
   std::list<access_record> m_access_log;
protected:
   void write_access_log(); 
};

class ring_tm_manager : public tm_manager 
{
public:
	ring_tm_manager( ptx_thread_info *thread, bool timing_mode );
   virtual ~ring_tm_manager();

   // interface for timing model validation and commit -- only for value-based tm
   virtual bool validate_addr( addr_t addr ); // validate a single word 
   virtual void commit_addr( addr_t addr ); // commit a single word 
   virtual void commit_core_side( ); // commit a transaction on the core side 

   void print(FILE *fout); 

protected:
   // tm policy specific code
   virtual void at_start(); 
   virtual bool at_access( memory_space *mem, bool potential_conflicting, bool rd, addr_t addr, int nbytes ); // return true if self-aborting 
   virtual void at_abort(); 
   virtual bool at_commit_validation(); // detect + resolve any conflicts: return true if self-aborting  
   virtual void at_commit_success(); // after conflicts are resolved 

   // ring TM metadata 
   int m_ring_starttime; 
   int m_ring_priority; // not used for now 
   addr_version_set_t m_read_word_version; 

   bool committer_abortee_conflict_resolution(); // return true if transaction is self-aborting 
   bool ring_tm_eager_conflict_resolution(bool rd); 
   void ring_tm_access(addr_t addr, int nbytes, bool rd); 
};

typedef unsigned long long tm_timestamp_t; 

class temporal_conflict_detector
{
public: 
   temporal_conflict_detector(); 
   virtual ~temporal_conflict_detector(); 

   // return with a timestamp indicating when the word can be last written 
   tm_timestamp_t at_transaction_read( addr_t addr ); 
   // update the last written time of a word when it is updated at commit 
   void update_word( addr_t addr, tm_timestamp_t new_time); 

   void dump(FILE *fp);

   static temporal_conflict_detector& get_singleton(); 

protected: 

   typedef std::unordered_map<addr_t, tm_timestamp_t> last_written_time_t; 
   last_written_time_t m_last_written_timetable; // a perfect record of when a word is last written 

   // a recency bloom filter for approximate timestamp storage 
   versioning_bloomfilter * m_rbloomfilter; 

   // function to standardize granularity of addresses 
   addr_t get_chunk_address( addr_t input_addr ); 

   // pointer to singleton 
   static temporal_conflict_detector * s_temporal_conflict_detector; 
}; 

class value_based_tm_manager : public tm_manager 
{
friend class tx_log_walker;
friend class tm_warp_info;
friend class shader_core_ctx;
friend class tm_parallel_pdom_warp_ctx_t;
public:
	value_based_tm_manager( ptx_thread_info *thread, bool timing_mode );
   virtual ~value_based_tm_manager();

   // interface for timing model validation and commit -- only for value-based tm
   virtual bool validate_addr( addr_t addr ); // validate a single word 
   virtual void commit_addr( addr_t addr ); // commit a single word 
   virtual void commit_core_side( ); // commit a transaction on the core side 
   virtual void validate_or_crash( ); // validate a transaction and crash if it is not valid

   // detect conflict between this transaction and the other 
   virtual bool has_conflict_with( tm_manager_inf * other_tx ); 

   // validate entire read-set (return true if pass)
   virtual bool validate_all( bool useTemporalCD ); 

   void dump_read_set(); 
   void dump_committed_set(); 

protected:
   // tm policy specific code
   virtual void at_start(); 
   virtual bool at_access( memory_space *mem, bool potential_conflicting, bool rd, addr_t addr, int nbytes ); // return true if self-aborting 
   virtual void at_abort(); 
   virtual bool at_commit_validation(); // detect + resolve any conflicts: return true if self-aborting  
   virtual void at_commit_success(); // after conflicts are resolved 

   void validate(); // perform conflict detection 

   // TM metadata 
   typedef tr1_hash_map<addr_t, unsigned int> addr_value_t;
   addr_value_t m_read_set_value; 
   bool m_violated; // true when this transaction is operating on inconsistent data 
   memory_space *m_gmem; // assume this is global memory 
   unsigned long long m_last_validation; // when was this transaction last validated 
   addr_set_t m_committed_set; 

   // temporal conflict detection 
   class temporal_cd_metadata {
   public:
      temporal_cd_metadata() { reset(); }
      void reset() {
         m_first_read_done = false; 
         m_first_read_time = 0; 
         m_last_writer_time = 0; 
      }
      void set_first_read_time( tm_timestamp_t new_time ) {
         assert(m_first_read_done == false); 
         m_first_read_time = new_time; 
         m_first_read_done = true; 
      }
      void update_last_written_time( tm_timestamp_t new_time ) {
         if (new_time > m_last_writer_time) {
            m_last_writer_time = new_time; 
         } 
      }
      bool conflict_exist() {
         if (m_first_read_done) {
            bool has_conflict = (m_last_writer_time > m_first_read_time); 
            return has_conflict; 
         } else {
            return false; 
         }
      }
      bool m_first_read_done; 
      tm_timestamp_t m_first_read_time; 
      tm_timestamp_t m_last_writer_time; // the last time when any word in the read-set is written 
   }; 
   temporal_cd_metadata m_temporal_cd_metadata; 

   // stats
   unsigned m_n_reread; 
   unsigned m_n_reread_violation; 
   unsigned m_n_timeout; 
   unsigned m_warp_level_raw; 
};

class tm_options : public OptionChecker
{
public:
   bool m_derive_done; 
   unsigned m_word_size; // size of a word 
   unsigned m_word_size_log2; 
   unsigned m_access_block_size; 
   unsigned m_access_block_size_log2; 
   unsigned m_nbank; // how many banks of memory/conflict tables are there

   bool m_lazy_conflict_detection; 
   bool m_check_bloomfilter_correctness; 
   bool m_abort_profile; 

   bool m_use_ring_tm; 
   bool m_ring_tm_eager_cd; 
   int m_ring_tm_size_limit; 
   bool m_ring_tm_bloomfilter; 
   int m_ring_tm_record_capacity; 
   bool m_ring_tm_version_read;
   int m_compressed_ring_capacity; // what each compressed ring structure appears to hold 
   int m_compressed_ring_size; // what it is really storing 
   bool m_ring_dump_record; 

   bool m_use_value_based_tm; 
   bool m_value_based_eager_cr; 
   int m_timeout_validation; // allow timeout validation

   bool m_enable_access_mode; // recognize intrinsics that specify if a access can potentially be conflicting 

   char *m_access_log_name; // name of the access log file 

   bool m_timing_mode_core_side_commit;

   bool m_exact_temporal_conflict_detection; 
   unsigned m_temporal_bloomfilter_size;  
   unsigned m_temporal_bloomfilter_n_hash;  
   unsigned m_temporal_cd_addr_granularity; 
   unsigned m_temporal_cd_addr_granularity_log2; 

   tm_options(); 
   virtual void check_n_derive(); 
   void reg_options(option_parser_t opp); 
private:
   option_parser_t m_opp; // retaining a handle of the option parser for option checking later 
};

struct conflict_set {
	bool empty() const { return m_tuids_have_read.empty() && m_tuids_have_written.empty(); }
   bool conflict() const; 
   bool read_only() const { return (!m_tuids_have_read.empty() && m_tuids_have_written.empty()); }
   bool write_only() const { return (m_tuids_have_read.empty() && !m_tuids_have_written.empty()); }
	tuid_set_t m_tuids_have_read;
	tuid_set_t m_tuids_have_written;
};

class tm_global_statistics {
public:
    unsigned long long m_n_aborts; 
    unsigned long long m_n_commits; 
    unsigned long long m_n_writing_commits; 
    unsigned long long m_n_transactions; 

    // number of aborts per transaction 
    pow2_histogram m_aborts_per_transaction; 
    pow2_histogram m_duration; // how long did the transaction run 
    pow2_histogram m_duration_first_rd; 

    unsigned m_concurrency;
    unsigned m_max_concurrency; 

    unsigned long long m_n_rollback_insn; 
    pow2_histogram m_n_insn_per_committed_txn;
    pow2_histogram m_n_insn_per_aborted_txn;
    unsigned long long m_n_insn_txn;

    unsigned long long m_n_intra_warp_aborts; 
    unsigned long long m_n_intra_block_aborts; 
    unsigned long long m_n_intra_core_aborts; 

    // conflict detection via signatures only 
    unsigned long long m_n_bloomfilter_detected_conflicts; 
    unsigned long long m_n_bloomfilter_false_conflicts; 

    // conflict detection via signatures against write set in address stream 
    unsigned long long m_n_bfxaddrstrm_detected_conflicts; 
    unsigned long long m_n_bfxaddrstrm_false_conflicts; 

    // conflict detection via hashed signatures of multiple threads against write set in address stream 
    unsigned long long m_n_bfxaddrstrm_hashed_detected_conflicts; 
    unsigned long long m_n_bfxaddrstrm_hashed_false_conflicts; 

    // workset size for committed transactions 
    pow2_histogram m_read_size; 
    pow2_histogram m_write_size; 
    pow2_histogram m_total_size; 

    // workset size for ALL transactions 
    pow2_histogram m_read_sz_all; 
    pow2_histogram m_write_sz_all; 
    pow2_histogram m_total_sz_all; 

    // workset size for transactions involved in false conflicts 
    pow2_histogram m_false_aborter_size; 
    pow2_histogram m_false_abortee_size; 

    pow2_histogram m_read_nblock; 
    pow2_histogram m_write_nblock; 
    pow2_histogram m_total_nblock; 
    linear_histogram m_total_nmem; 

    pow2_histogram m_num_thread_conflict_per_commit; 
    linear_histogram m_num_core_conflict_per_commit; 
    linear_histogram m_num_core_bfxaddrstrm_per_commit; 
    linear_histogram m_num_core_bfsum_per_commit; 
    linear_histogram m_num_core_bfsum_extra_per_commit; 
    linear_histogram m_num_core_bfhash_per_commit; 

    pow2_histogram m_num_thread_match_per_bfhash_match; 

    // traffic if a coherence protocol is below the TM 
    unsigned long long m_lazy_coherence_traffic_thread_level; 
    unsigned long long m_lazy_coherence_traffic_core_level; 
    pow2_histogram m_lazy_coherence_traffic; 

    // profiling read access to transactional data 
    pow2_histogram m_raw_footprint; 
    pow2_histogram m_raw_access; 

    pow2_histogram m_conflict_footprint; 
    int m_max_conflict_footprint; 

    pow2_histogram m_transaction_footprint; 
    size_t m_max_transaction_footprint; 

    pow2_histogram m_readonly_footprint; 
    int m_max_readonly_footprint; 

    pow2_histogram m_writeonly_footprint; 
    int m_max_writeonly_footprint; 

    void sample_footprint(); 

    // # abort caused by each memory address 
    tr1_hash_map<addr_t, unsigned int> m_abort_profile; 
    void record_abort_at_address(addr_t addr, const conflict_set& cs); 
    void dump_abort_profile(FILE *csv); 

    pow2_histogram m_n_reread;
    unsigned m_n_reread_violation; 

    pow2_histogram m_write_buffer_footprint;

    pow2_histogram m_n_rewrite; 
    pow2_histogram m_n_read; 
    pow2_histogram m_n_write; 

    pow2_histogram m_n_timeout_validation;

    // TM reg states
    linear_histogram m_regs_modified_max;
    linear_histogram m_regs_buffered_max;
    linear_histogram m_regs_read_max;

    // intra-warp conflict detected and aborted 
    unsigned long long m_n_intra_warp_detected_conflicts; 
    unsigned long long m_n_vcd_tcd_mismatch; 

    // number of read-after-write access between threads within a warp 
    pow2_histogram m_n_warp_level_raw;

    void inc_concurrency() {
        m_concurrency += 1; 
        assert(m_concurrency >= 0); 
        m_max_concurrency = std::max(m_concurrency, m_max_concurrency); 
    }

    void dec_concurrency() {
        m_concurrency -= 1; 
        assert(m_concurrency >= 0); 
        m_max_concurrency = std::max(m_concurrency, m_max_concurrency); 
    }

    void record_commit_tx_size(size_t read_size, size_t write_size, size_t access_size) {
 //       assert(access_size <= read_size + write_size); // 2015-11-18
        m_read_size.add2bin(read_size); 
        m_write_size.add2bin(write_size); 
        m_total_size.add2bin(access_size); 
        m_read_sz_all.add2bin(read_size); 
        m_write_sz_all.add2bin(write_size); 
        m_total_sz_all.add2bin(access_size); 
    }

    void record_abort_tx_size(size_t read_size, size_t write_size, size_t access_size) {
//        assert(access_size <= read_size + write_size); // Doesn't hold after 2015-09-21 ugly hack !
        m_read_sz_all.add2bin(read_size); 
        m_write_sz_all.add2bin(write_size); 
        m_total_sz_all.add2bin(access_size); 
    }

    void record_false_conflict_info(size_t aborter_size, size_t abortee_size) {
        m_false_aborter_size.add2bin(aborter_size); 
        m_false_abortee_size.add2bin(abortee_size); 
    }

    void record_tx_blockcount(const addr_set_t& read_set_block, 
                              const addr_set_t& write_set_block, 
                              const addr_set_t& access_set_block);

    void record_raw_info(size_t footprint, unsigned n_accesses) {
        m_raw_footprint.add2bin(footprint); 
        m_raw_access.add2bin(n_accesses); 
    }

    tm_global_statistics() 
      : m_n_aborts(0), 
        m_n_commits(0), 
        m_n_writing_commits(0), 
        m_n_transactions(0), 
        m_aborts_per_transaction("tm_aborts_per_transactions"), 
        m_duration("duration"), 
        m_duration_first_rd("duration_first_rd"), 
        m_concurrency(0), 
        m_max_concurrency(0), 
        m_n_rollback_insn(0),
        m_n_insn_per_committed_txn("tm_n_insn_per_committed_txn"),
        m_n_insn_per_aborted_txn("tm_n_insn_per_aborted_txn"),
        m_n_insn_txn(0),
        m_n_intra_warp_aborts(0), 
        m_n_intra_block_aborts(0), 
        m_n_intra_core_aborts(0), 
        m_n_bloomfilter_detected_conflicts(0), 
        m_n_bloomfilter_false_conflicts(0), 
        m_n_bfxaddrstrm_detected_conflicts(0), 
        m_n_bfxaddrstrm_false_conflicts(0), 
        m_n_bfxaddrstrm_hashed_detected_conflicts(0), 
        m_n_bfxaddrstrm_hashed_false_conflicts(0), 
        m_read_size("tm_read_size"),
        m_write_size("tm_write_size"),
        m_total_size("tm_total_size"), 
        m_read_sz_all("tm_read_sz_all"),
        m_write_sz_all("tm_write_sz_all"),
        m_total_sz_all("tm_total_sz_all"), 
        m_false_aborter_size("tm_false_aborter_size"),
        m_false_abortee_size("tm_false_abortee_size"),
        m_read_nblock("tm_read_nblock"),
        m_write_nblock("tm_write_nblock"),
        m_total_nblock("tm_total_nblock"), 
        m_total_nmem(1, "tm_total_nmem"), 
        m_num_thread_conflict_per_commit("tm_n_threads_conflict_per_commit"), 
        m_num_core_conflict_per_commit(1, "tm_n_core_conflict_per_commit"), 
        m_num_core_bfxaddrstrm_per_commit(1, "tm_n_core_bfxaddrstrm_per_commit"), 
        m_num_core_bfsum_per_commit(1, "tm_n_core_bfsum_per_commit"), 
        m_num_core_bfsum_extra_per_commit(1, "tm_n_core_bfsum_extra_per_commit"), 
        m_num_core_bfhash_per_commit(1, "tm_num_core_bfhash_per_commit"),
        m_num_thread_match_per_bfhash_match("tm_n_thread_match_per_bfhash_match"),
        m_lazy_coherence_traffic_thread_level(0), 
        m_lazy_coherence_traffic_core_level(0), 
        m_lazy_coherence_traffic("tm_lazy_coherence_traffic"),  
        m_raw_footprint("tm_raw_footprint"), 
        m_raw_access("tm_raw_access"),
        m_conflict_footprint("tm_conflict_footprint"), 
        m_max_conflict_footprint(0),
        m_transaction_footprint("tm_transaction_footprint"), 
        m_max_transaction_footprint(0),
        m_readonly_footprint("tm_readonly_footprint"), 
        m_max_readonly_footprint(0),
        m_writeonly_footprint("tm_writeonly_footprint"), 
        m_max_writeonly_footprint(0),
        m_n_reread("tm_n_reread"),
        m_n_reread_violation(0),
        m_write_buffer_footprint("tm_write_buffer_footprint"),
        m_n_rewrite("tm_n_rewrite"), 
        m_n_read("tm_n_read"),
        m_n_write("tm_n_write"),
        m_n_timeout_validation("tm_n_timeout_validation"),
        m_regs_modified_max(1, "m_regs_modified_max"),
        m_regs_buffered_max(1, "m_regs_buffered_max"),
        m_regs_read_max(1, "m_regs_read_max"), 
        m_n_intra_warp_detected_conflicts(0), 
        m_n_vcd_tcd_mismatch(0), 
        m_n_warp_level_raw("tm_n_warp_level_raw")
    { }

    void print(FILE *fout); 
    void print_short(FILE *fout); 
    void visualizer_print( gzFile visualizer_file ); 
};
#define TM_MSG_INV 100

// unified bloomfilter configuration for all threads 
class tm_bloomfilter_options 
{
public: 
   bool m_init; 
   unsigned int m_size; 
   unsigned int m_n_hashes; 
   std::vector<int> m_funct_ids; 

   tm_bloomfilter_options(); 
   void reg_options(option_parser_t opp); // register commandline options 
   void init(); // initialize bloomfilter configurations 
};

// read/write set of a transaction as represented by bloomfilter signatures  
class tm_bloomfilter_core_set;
class tm_bloomfilter_set {
public:
   tm_bloomfilter_set(bool counter_based = false); 

   void mem_access(bool rd, unsigned addr, unsigned nbytes); 

   // match against a whole signature from another transaction
   bool match_access_set(const tm_bloomfilter_set& other); 

   // match against a set of addresses from another transaction 
   bool match_access_set(const addr_set_t& w_set); 

   void clear(); // called at abort (and commit?)

   void set_core_set(int core_id, tm_bloomfilter_core_set *core_bf) {
      assert(m_core_bf == NULL or m_core_bf == core_bf); 
      m_core_bf = core_bf;
      m_core_id = core_id; 
   }
protected:
   bloomfilter m_access_set; // read + write 
   bloomfilter m_write_set; 

   tm_bloomfilter_core_set* m_core_bf; 
   int m_core_id; 
};

class tm_bloomfilter_core_set : public tm_bloomfilter_set {
public:
   tm_bloomfilter_core_set(); 

   // updating the counting bloomfilter 
   void update_access_bloomfilter(const std::vector<int>& update_positions); 

   // removing the contribution of a particular thread's access set 
   void remove_access_set(const bloomfilter &thread_access_set); 

   void set_core_set(tm_bloomfilter_core_set *core_bf) { assert(0); }
};

// track the access sets of all threads in a core 
class tm_bloomfilter_hashed_core_set : public tm_bloomfilter_set {
public:
   tm_bloomfilter_hashed_core_set(); 
   static void reg_options(option_parser_t opp); // register commandline options 
   void set_core_id(int core_id) { assert(m_core_id == core_id || m_core_id == -1); m_core_id = core_id; } 

   void mem_access(int hw_thread_id, bool rd, unsigned addr, unsigned nbytes); 

   // match against a set of addresses from another transaction 
   // return the number of threads that match with this set and the matched thread vector  
   unsigned int match_access_set(const addr_set_t& w_set, hashtable_bits_mt::tvec_t &matched_threads); 

   void clear(int hw_thread_id); // called at abort (and commit?)
   void clear_all(); // called at kernel launch? 

protected:
   static unsigned int s_thread_hash_size; 
   bloomfilter_mt m_access_set_mt; // read + write 
   int m_core_id; 
   static const bool s_non_hash_match = false; 
};

class tm_global_state {
public:	
   tm_global_state() : m_conflict_footprint(0), m_readonly_footprint(0), m_writeonly_footprint(0) { } 
	tuid_set_t mem_access( unsigned tuid, bool rd, unsigned addr, unsigned nbytes );
	void remove_tuid( unsigned tuid, const addr_set_t &read_set, const addr_set_t &write_set );
	void register_thread( ptx_thread_info *thrd );
	void unregister_thread( ptx_thread_info *thrd );
	ptx_thread_info *lookup_tuid(unsigned tuid);

   // runs through the conflict vector for the whole write_set and return with the set of conflicting tx
   tuid_set_t lazy_conflict_detection( const addr_set_t& write_set ); 

   void memaccess_tx_bf( unsigned tuid, bool rd, unsigned addr, unsigned nbytes ); 
   void commit_tx_bf( unsigned tuid, const addr_set_t& write_set, const tuid_set_t& true_conflict_set );
   void abort_tx_bf( unsigned tuid ); 

   void estimate_coherence_traffic(const addr_set_t& write_set, unsigned commit_tuid); 

   void print_resource_usage( FILE *fout ); 
   const tm_bloomfilter_set& get_BF(unsigned tuid) const;

   int conflict_footprint() const { return m_conflict_footprint; }
private:
	typedef tr1_hash_map<unsigned,conflict_set> conflict_mem_t;
	conflict_mem_t m_mem;
   int m_conflict_footprint; 
   int m_readonly_footprint; 
   int m_writeonly_footprint; 

	typedef tr1_hash_map<unsigned,ptx_thread_info*> tuid_to_thread_t;
	tuid_to_thread_t m_tlookup;

	typedef tr1_hash_map<unsigned,tm_bloomfilter_set> bloomfilter_access_set_t;
	bloomfilter_access_set_t m_bfaccess;

   // the bloomfilter that summarize the access of threads in each core 
   typedef tr1_hash_map<unsigned, tm_bloomfilter_core_set> core_bloomfilter_t; 
   core_bloomfilter_t m_bfcore; 

   // the bloomfilter that summarize the access of threads in each core (version 2)
   typedef tr1_hash_map<unsigned, tm_bloomfilter_hashed_core_set> core_hash_bloomfilter_t; 
   core_hash_bloomfilter_t m_bfcorehash; 
   tr1_hash_map<unsigned, hashtable_bits_mt::tvec_t> m_threads_in_tx; 
};

///////////////////////////////////////////////////////////////////////////////
// TM ring conflict detection implementation 

namespace tm_ring {
enum status_t {
   INVALID, 
   WRITING, 
   COMPLETE
}; 
}

class tm_ring_commit_record
{
public: 
   tm_ring_commit_record(unsigned tuid, const addr_set_t& write_set, int commit_time); 

   // modifiers 
   void set_status(enum tm_ring::status_t new_status) { m_status = new_status; } 

   // accessor 
   enum tm_ring::status_t status() const { return m_status; }
   int commit_time() const { return m_timestamp; }
   bool match(const addr_version_set_t& other_tx_read_set) const; // match against a given read set 
   bool match_filter(const addr_version_set_t& other_tx_read_set) const; // match the write filter against a given read set 

   void print(FILE *fout) const; 
protected:
   unsigned m_tuid; 
   int m_timestamp; 
   addr_set_t m_write_set; // for ideal conflict detection 
   int m_priority; 
   enum tm_ring::status_t m_status; 
   bloomfilter m_write_filter; // for bloom filter based conflict detection 
}; 

class tm_ring_compressed_record 
{
public:
   tm_ring_compressed_record(int base_index, unsigned int n_records, unsigned int compressed_size); 

   // push a single commit record into the structure 
   void commit_record(const addr_set_t& write_set, int commit_time); 

   // match records inside this structure against this read set (with version tags to limit scope of matching)
   bool match(const addr_version_set_t& other_tx_read_set, int starttime) const;
protected:
   int m_base_index; 
   int m_next_rec_index; 
   unsigned int m_n_records; 
   unsigned int m_compressed_size; 
   bloomfilter_mt m_write_filter; // for hashed bloom filter array based conflict detection 
};

#include <deque> 

class tm_ring_global
{
public:
   tm_ring_global(); 
   
   // modifier
   void commit_tx(unsigned tuid, const addr_set_t& write_set); // commit transaction to ring 

   // accessor 
   int ring_index() const { return m_ring_index; }

   // check if any of the commit records from ring index downto start_time + 1 conflicts with given read set 
   // return start_time if there is no conflict 
   int check_conflict(const addr_version_set_t& other_tx_read_set, int start_time) const; 

   void print(FILE *fout) const; 

private: 
   typedef std::deque<tm_ring_commit_record> ring_t; 
   ring_t m_ring; 
   int m_ring_index; // the newest ring entry

   typedef std::deque<tm_ring_compressed_record> compressed_ring_t; 
   compressed_ring_t m_compressed_ring; 

   // modifier
   void commit_tx_single_record(unsigned tuid, const addr_set_t& write_set); // commit transaction to ring 
}; 

// ring tm stats 
class tm_ring_stats
{
public:
   pow2_histogram m_commit_rec_distance; // #record checked at each conflict checking 
   pow2_histogram m_commit_rec_dist_conflict; // #record checked with a conflict hit 
   pow2_histogram m_commit_rec_dist_no_conflict; // #record checked with a conflict miss
   pow2_histogram m_commit_rec_actual_conflict_distance; // the actual commit record that hit (in distance from ring_index)
   unsigned int m_ring_bloomfilter_false_conflicts; 

   tm_ring_stats() 
      : m_commit_rec_distance("tm_commit_rec_distance"), 
        m_commit_rec_dist_conflict("tm_commit_rec_dist_conflict"), 
        m_commit_rec_dist_no_conflict("tm_commit_rec_dist_no_conflict"), 
        m_commit_rec_actual_conflict_distance("tm_commit_rec_actual_conflict_distance"),
        m_ring_bloomfilter_false_conflicts(0)
   { }

   void print(FILE *fout); 
};

#endif



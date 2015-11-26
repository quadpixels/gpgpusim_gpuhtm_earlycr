/***************************************** 
 
  tm_manager.cc
 
  encapsulates data required by transactional memory
  includes restore point, memory values, functionality
  for manipulating ptx_thread and tm state
 
 
****************************************/

#include "tm_manager.h"
#include "tm_manager_internal.h"
#include "../gpgpu-sim/shader.h"

extern unsigned long long gpu_sim_cycle; // for data collection ><
extern unsigned long long gpu_tot_sim_cycle; // for data collection ><
extern void CartographerTimeSeries_printReadSet(unsigned int cid);

tm_global_state g_tm_global_state;
tm_ring_global g_tm_ring_global; 
tm_global_statistics g_tm_global_statistics; 
tm_ring_stats g_tm_ring_stats; 
tm_options g_tm_options; 
tm_bloomfilter_options g_tm_bloomfilter_options; 

static const unsigned g_watched_thread_a = 13752; // 668 + 46080 + 1; 
static const unsigned g_watched_thread_b = -1; // 1469 + 46080 + 1; 
static const unsigned g_watched_thread_c = -1; // 967 + 46080 + 1;

extern unsigned g_watched_tid, g_num_stagger_aborts, g_num_staggers, g_num_stagger_samewarp;

extern void Cartographer_onTMManagerStart(unsigned uid);
extern void Cartographer_onTMManagerAbort(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi);
extern void Cartographer_onTMManagerCommit(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi);

///////////////////////////////////////////////////////////////////////////////

tm_manager_inf* tm_manager_inf::create_tm_manager( ptx_thread_info *thread, bool timing_mode )
{
   // g_tm_options.check_n_derive_options(); //HACK: ensure that the derived options are calculated

   tm_manager_inf* new_tx;
   if (g_tm_options.m_use_value_based_tm) {
      new_tx = new value_based_tm_manager(thread, timing_mode); 
   } else if (g_tm_options.m_use_ring_tm) {
      new_tx = new ring_tm_manager(thread, timing_mode); 
   } else {
      new_tx = new tm_manager(thread, timing_mode); 
   }

   return new_tx; 
}

unsigned g_tm_manager_next_uid = 0;

tm_manager_inf::tm_manager_inf( ptx_thread_info *thread, bool timing_mode )
   : m_uid(++g_tm_manager_next_uid), m_timing_mode(timing_mode), 
     m_nesting_level(0), m_thread(thread),
     m_thread_uid(m_thread->get_uid()), 
     m_thread_sc(m_thread->get_hw_sid()),
     m_thread_hwtid(m_thread->get_hw_tid()),
     m_ref_count(1), m_abort_count(0), 
     m_is_warp_level(false) 
{ }

bool tm_manager_inf::watched() const 
{
   #if 0
   return (m_thread_uid == g_watched_thread_a 
           or m_thread_uid == g_watched_thread_b
           or m_thread_uid == g_watched_thread_c);
   #else 
   if (m_thread_uid == g_watched_tid) return true; else return false;
   #endif
}

int tm_manager_inf::inc_ref_count() { m_ref_count++; return m_ref_count; }
int tm_manager_inf::dec_ref_count() { m_ref_count--; assert(m_ref_count >= 0); return m_ref_count; }
int tm_manager_inf::get_ref_count() { return m_ref_count; }

tm_manager_inf::~tm_manager_inf() 
{
   assert(m_ref_count == 0); // ensure all reference to it are gone before deleting the object
}

///////////////////////////////////////////////////////////////////////////////

tm_manager::tm_manager(	ptx_thread_info *thread, bool timing_mode ) 
   : tm_manager_inf(thread, timing_mode)
{
	m_abort_count = 0;
	m_nesting_level = 0;
	m_stats = trans_stats();
   m_raw_access = 0; 
   m_read_conflict_detection = true; 
   m_write_conflict_detection = true; 
   m_version_management = true; 
   m_start_cycle = -1; 
   m_first_read_cycle = -1; 
   m_n_read = 0;
   m_n_write = 0; 
   m_n_rewrite = 0;
   m_gmem_view_tx = NULL; 
}

tm_manager::~tm_manager()
{ }

void tm_manager::at_start() 
{ 
   g_tm_global_state.register_thread(m_thread);
}

void tm_manager::start()
{
	if( !m_thread->is_in_transaction() ) {
		m_thread->start_transaction(this);
		m_nesting_level = 1;
		m_start_cycle = gpu_sim_cycle;
		g_tm_global_statistics.m_n_transactions += 1; 
		g_tm_global_statistics.inc_concurrency();

		// policy specific start transaction code
		at_start();
		Cartographer_onTMManagerStart(m_uid);

#ifdef DEBUG_TM
		dim3 tid = m_thread->get_tid();
		dim3 ctaid = m_thread->get_ctaid();
		printf("[%Lu] starting transaction %u (uid=%u) for tid=(%u,%u,%u) cta=(%u,%u,%u) tuid=%u, sc=%u, hwtid=%u\n", 
		   gpu_sim_cycle,
		   m_thread->tm_num_transactions(),
		   m_uid,
		   tid.x, tid.y, tid.z, ctaid.x, ctaid.y, ctaid.z, 
		   m_thread_uid, m_thread_sc, m_thread_hwtid );
#else
      if (g_tm_global_statistics.m_n_transactions % TM_MSG_INV == 0) {
         g_tm_global_statistics.print_short(stdout); 
      }
#endif
	} else {
		m_nesting_level ++;
	}
}

bool ring_tm_manager::ring_tm_eager_conflict_resolution(bool rd)
{
   if (g_tm_options.m_ring_tm_eager_cd) {
      if (rd == true) {
         if (m_ring_starttime == -1) {
            m_ring_starttime = g_tm_ring_global.ring_index(); 
         } else {
            // eagerly checking for conflict with the commit records (and no need to check them later)
            int conflicting_record = g_tm_ring_global.check_conflict(m_read_word_version, m_ring_starttime); 
            if (conflicting_record > m_ring_starttime) {
               // m_ring_starttime = g_tm_ring_global.ring_index(); // grab a new ring index 
               m_ring_starttime = -1; // just reset the start time 
               // abort tx (by returning true) as it is conflicting with previously committed tx 
               return true; 
            } else {
               m_ring_starttime = g_tm_ring_global.ring_index(); 
            }
         }
      }
   } else {
      // RING_TM: grab the most update ring index if this is the first read in the transaction 
      if (rd == true and m_ring_starttime == -1) {
         m_ring_starttime = g_tm_ring_global.ring_index(); 
      }
   }
   return false; 
}

void ring_tm_manager::ring_tm_access(addr_t addr, int nbytes, bool rd)
{
   if (g_tm_options.m_use_ring_tm == false or rd == false) return; 
   unsigned word_size_log2 = g_tm_options.m_word_size_log2; 
   for( addr_t a=addr; a < addr+nbytes; ++a ) {
      addr_t waddr = a >> word_size_log2;
      int ring_time = (g_tm_options.m_ring_tm_version_read)? g_tm_ring_global.ring_index() : 0; 
      if (m_read_word_version.find(waddr) != m_read_word_version.end()) {
         m_read_word_version[waddr] = std::min(m_read_word_version[waddr], ring_time);
      } else {
         m_read_word_version[waddr] = ring_time; 
      }
   }
}

// update read/write sets in different granularities 
void tm_manager::update_access_sets(addr_t addr, unsigned nbytes, bool rd, bool detect_conflict) 
{
   unsigned word_size_log2 = g_tm_options.m_word_size_log2; 
   unsigned block_size_log2 = g_tm_options.m_access_block_size_log2; 
   for( unsigned a=addr; a < addr+nbytes; ++a ) {
      addr_t word_addr = a >> word_size_log2; 
      addr_t block_addr = a >> block_size_log2; 
      if (detect_conflict) {
         m_access_word_set.insert(word_addr); 
         m_access_block_set.insert(block_addr); 
         if( rd ) {
            m_read_set.insert(a);
            m_read_word_set.insert(word_addr); 
            m_read_block_set.insert(block_addr); 
         } else {
            m_write_set.insert(a);
            m_write_word_set.insert(word_addr); 
            m_write_block_set.insert(block_addr); 
            if (m_gmem_view_tx != NULL) {
               m_gmem_view_tx->m_warp_level_write_word_set.insert(word_addr); 
            }
         }
      }
   }
   // buffered write footprint and rewrite bandwidth  
   if (!rd) {
      unsigned word_size = g_tm_options.m_word_size; 
      for( unsigned a=addr; a < addr+nbytes; a+=word_size ) { // traversing in words to avoid self-overlap
         addr_t word_addr = a >> word_size_log2; 
         if (m_buffered_write_word_set.find(word_addr) != m_buffered_write_word_set.end()) {
            m_n_rewrite += 1; // this is rewrite in bytes 
         } else {
            m_buffered_write_word_set.insert(word_addr); 
         }
      }
   }
   // bandwidth tracking 
   addr_t nwords = nbytes >> word_size_log2;
   if (rd) {
	   m_n_read_all += nwords;
      if (detect_conflict) m_n_read += nwords;
   } else {
      m_n_write += nwords; 
   }
}

// do eager conflict detection: update conflict tx set 
bool tm_manager::at_access( memory_space *mem, bool potential_conflicting, bool rd, addr_t addr, int nbytes )
{
	if( !m_timing_mode and potential_conflicting) {
		// detect conflicts
		tuid_set_t conflicts = g_tm_global_state.mem_access( m_thread_uid, rd, addr, nbytes );
		m_conflict_tuids.insert( conflicts.begin(), conflicts.end() );
		for( tuid_set_t::iterator t=conflicts.begin(); t!=conflicts.end();++t ) {
			unsigned tuid = *t;
			ptx_thread_info *thd = g_tm_global_state.lookup_tuid(tuid);
			tm_manager *tm = dynamic_cast<tm_manager*>( thd->get_tm_manager() );
			assert( tm != NULL ); // or should have deregistered itself already...
			tm->m_conflict_tuids.insert( m_thread_uid );
		}
	}
   return false; 
}

bool tm_manager::tm_access( memory_space *mem, memory_space_t space, bool rd, addr_t addr, void *vp, int nbytes, tm_access_uarch_info& uarch_info )
{
   // limit conflict detection to memory space that can have data race 
   bool potential_conflicting; 
   switch (space.get_type()) {
      case param_space_local:
      case local_space: 
         assert(m_thread->is_local_memory_space(mem) == true); 
      case const_space: 
      case tex_space: 
      case param_space_kernel:
      case shared_space: // HACK: no tracking conflicts for shared memory 
         potential_conflicting = false; 
         break; 
      default:
         potential_conflicting = true; 
         break; 
   }

   // access mode control -- further narrow down the detecting conflict ones
   bool detecting_conflict = potential_conflicting; 
   if (g_tm_options.m_enable_access_mode) {
      if (rd) {
         if (m_read_conflict_detection == false) 
            detecting_conflict = false; 
      } else {
         if (m_write_conflict_detection == false) 
            detecting_conflict = false; 
      }
   }

   // update uarch info
   uarch_info.m_conflict_detect = detecting_conflict; 
   uarch_info.m_version_managed = m_version_management and potential_conflicting; 

   // tm policy specific code 
   bool self_abort = at_access(mem, detecting_conflict, rd, addr, nbytes); 
   if (self_abort) {
      abort();
      uarch_info.m_timeout_validation_fail.set(m_thread_hwtid % tm_access_uarch_info::warp_size); 
      return false; // terminate the access if the transaction has self-aborted 
   }

	if (potential_conflicting) {
      // update read/write sets  
      update_access_sets(addr, nbytes, rd, detecting_conflict); 

      if (detecting_conflict == true and rd == true and m_first_read_cycle == -1) {
         m_first_read_cycle = gpu_sim_cycle; 
      }
	}

	// Update read write stats
   for( unsigned a=addr; a < addr+nbytes; ++a ) {
      unsigned temp = (a&((1<<CACHE_LINE_SIZE)-1));
      m_stats.m_cache_lines_accessed.insert(temp);
   }

	// verify memory accesses are aligned to smaller power of 2 than bucket size
	assert( (addr>>TM_MEM_BUCKET_SIZE) == ((addr+nbytes-1) >> TM_MEM_BUCKET_SIZE) );

	// access transaction's view of memory...
   tm_mem_t* spec_mem = &m_tm_mem; 
   if (m_gmem_view_tx != NULL and space == global_space) {
      spec_mem = &(m_gmem_view_tx->m_tm_mem); 
   }
	if( rd ) {
      bool tx_raw_access = false; 
		tm_mem_t::iterator i = spec_mem->find(mem);
		if( i != spec_mem->end() ) {
			// we have written to this memory space
			tm_mem_hash_t &hash = i->second;
			tm_mem_hash_t::iterator i =	hash.find(addr>>TM_MEM_BUCKET_SIZE);
			if( i != hash.end() ) {
				// we have written to a location close to this address
				tm_mem_bucket &bucket = i->second;
				unsigned a = (addr&((1<<TM_MEM_BUCKET_SIZE)-1));

				assert( (1<<TM_MEM_BUCKET_SIZE) == 8 * sizeof(unsigned long long) ); // code below expects this
				unsigned long mask = ((unsigned long long)1)<<a;
				assert((a + nbytes-1)<(1<<TM_MEM_BUCKET_SIZE)); // Make sure the data being written doesnt exceed bucket size
				for( unsigned n=0; n < nbytes; ++n, mask <<= 1 ) {
					if( (bucket.m_modified & mask) != 0 ) {
						((unsigned char*)vp)[n] = bucket.m_data[a + n];
                        if (potential_conflicting) {
                           m_raw_set.insert(a + n); // record that this bit is read after written
                           m_raw_access++;
                           tx_raw_access = true;
                        }
					} else {
						mem->read(addr + n, 1, ((char*)vp)+n);
                    }
				}
			} else
				mem->read(addr, nbytes, vp);
		} else 
			mem->read(addr, nbytes, vp);
      if (tx_raw_access) {
         uarch_info.m_writelog_access.set(m_thread_hwtid % tm_access_uarch_info::warp_size); 
      }
      if (watched()) {
         printf("[TMM-%llu] Thd %u txload[%#08x]=%#x, nbytes=%d, RSSize=%lu, RWSSize=%lu\n", gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid,
             addr, *((unsigned int*)vp), nbytes, this->m_read_set.size(), this->m_read_word_set.size());
      }
	} else {
		tm_mem_bucket &bucket = (*spec_mem)[mem][addr>>TM_MEM_BUCKET_SIZE];
		unsigned a = (addr&((1<<TM_MEM_BUCKET_SIZE)-1));
		unsigned long mask = ((unsigned long long)1)<<a;
		assert((a + nbytes-1)<(1<<TM_MEM_BUCKET_SIZE));// Make sure the data being written doesnt exceed bucket size
		for( unsigned n=0; n < nbytes; ++n, mask <<= 1 ) {
			bucket.m_modified |= mask;
			bucket.m_data[a + n] = ((unsigned char*)vp)[n];
		}
      if (watched()) {
         printf("[TMM-%llu] Thd %u txstore[%#08x]=%#x ()\n", gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid,
             addr, *((unsigned int*)vp));
      }
	}

	// buffer updates 
	if( !rd ) 
		m_write_data.push_back( access_record(this,mem,rd,addr,vp,nbytes) );

   // append access log 
   if (potential_conflicting and g_tm_options.m_access_log_name != NULL) {
      m_access_log.push_back( access_record(this,mem,rd,addr,vp,nbytes) ); 
   }

   return true;
}

void tm_manager::at_abort()
{
   g_tm_global_state.abort_tx_bf(m_thread_uid); 
	g_tm_global_state.remove_tuid(m_thread_uid,m_read_set,m_write_set);
	m_conflict_tuids.clear();
}

void tm_manager::abort()
{
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_thread->m_core);
	struct tm_warp_info* twi = NULL;
	if (scc) { twi = &(scc->m_warp[m_thread->m_hw_wid].get_tm_warp_info()); }
	Cartographer_onTMManagerAbort(m_uid, this, twi);
	m_thread->m_is_committing = false;
    at_abort();
	g_tm_global_statistics.m_n_aborts += 1; 
    g_tm_global_statistics.record_abort_tx_size(m_read_word_set.size(), m_write_word_set.size(), m_access_word_set.size());
    g_tm_global_statistics.record_raw_info(m_raw_set.size(), m_raw_access);

	m_stats.m_abort =1;
	m_abort_count++;
	write_stats();
	m_thread->tm_rollback();
	m_read_set.clear();
	m_write_set.clear();
	m_read_word_set.clear();
	m_write_word_set.clear();
	m_access_word_set.clear();
	m_read_block_set.clear();
	m_write_block_set.clear();
	m_access_block_set.clear();
	m_warp_level_write_word_set.clear();
	m_buffered_write_word_set.clear();
	m_raw_set.clear();
	m_raw_access = 0;
	m_n_read = 0;
	m_n_write = 0;
	m_n_rewrite = 0;
	m_tm_mem.clear();
	m_write_data.clear();
	m_access_log.clear();
	m_nesting_level=1;
	m_start_cycle = gpu_sim_cycle;

   // reset access mode 
   m_read_conflict_detection = true; 
   m_write_conflict_detection = true; 
   m_version_management = true; 

   if (watched()) {
      printf("[TMM-%llu] Thd %u Aborts, PC=%x\n", gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid,
    	m_thread->get_pc());
   }

#ifdef DEBUG_TM
	dim3 tid = m_thread->get_tid();
	dim3 ctaid = m_thread->get_ctaid();
	printf("[%Lu]    -> ABORT transaction %u (uid=%u) for tid=(%u,%u,%u) cta=(%u,%u,%u) tuid=%u, sc=%u, hwtid=%u (#aborts=%u)\n",
		   gpu_sim_cycle,
		    m_thread->tm_num_transactions(), m_uid,
			tid.x, tid.y, tid.z, ctaid.x, ctaid.y, ctaid.z, 
			m_thread_uid, m_thread_sc, m_thread_hwtid, m_abort_count );
#else
   if (g_tm_global_statistics.m_n_aborts % TM_MSG_INV == 0) {
      g_tm_global_statistics.print_short(stdout); 
   }
#endif
}

void tm_manager::add_rollback_insn( unsigned insn_count )
{
   g_tm_global_statistics.m_n_rollback_insn += insn_count;
   g_tm_global_statistics.m_n_insn_txn += insn_count;
   g_tm_global_statistics.m_n_insn_per_aborted_txn.add2bin(insn_count);
}

void tm_manager::add_committed_insn( unsigned insn_count )
{
   g_tm_global_statistics.m_n_insn_per_committed_txn.add2bin(insn_count);
   g_tm_global_statistics.m_n_insn_txn += insn_count;
}

void tm_manager::committer_win_conflict_resolution()
{
   // obtain a set of conflicting threads depending on the detection mechanism 
   const tuid_set_t *conflict_tuids = &m_conflict_tuids; 
   tuid_set_t lazy_conflict_tuids; 
   if (g_tm_options.m_lazy_conflict_detection) {
      lazy_conflict_tuids = g_tm_global_state.lazy_conflict_detection(m_write_set); 
      conflict_tuids = &lazy_conflict_tuids; 
   }

   g_tm_global_state.commit_tx_bf(m_thread_uid, m_write_word_set, *conflict_tuids); 

   // obtain runtime info about the commiting transaction 
   ptx_thread_info *commit_thread = g_tm_global_state.lookup_tuid(m_thread_uid);
   unsigned commit_core_id = commit_thread->get_hw_sid(); 
   unsigned commit_block_id = commit_thread->get_hw_ctaid(); 
   unsigned commit_warp_id = commit_thread->get_hw_wid(); 

   // track the cores and #threads involve in the conflict 
   std::bitset<32> conflicting_cores; 
   conflicting_cores.reset();
   unsigned int n_conflicting_threads = 0; 

   for( tuid_set_t::const_iterator t=conflict_tuids->begin(); t!= conflict_tuids->end(); ++t ) {
      unsigned tuid = *t;
      ptx_thread_info *ct = g_tm_global_state.lookup_tuid(tuid);
      if( ct != NULL ) {
         if( ct->get_uid() != m_thread_uid ) {
            unsigned conflict_core_id = ct->get_hw_sid(); 
            unsigned conflict_block_id = ct->get_hw_ctaid(); 
            unsigned conflict_warp_id = ct->get_hw_wid(); 
            conflicting_cores.set(conflict_core_id); 
            if (conflict_core_id == commit_core_id) {
               g_tm_global_statistics.m_n_intra_core_aborts += 1; 
               if (conflict_block_id == commit_block_id) {
                  g_tm_global_statistics.m_n_intra_block_aborts += 1; 
               }
               if (conflict_warp_id == commit_warp_id) {
                  g_tm_global_statistics.m_n_intra_warp_aborts += 1; 
               }
            }
            n_conflicting_threads += 1; 
            // thread ct is still in a transaction...
            tm_manager *ct_tm = dynamic_cast<tm_manager*>( ct->get_tm_manager() );
            assert( ct_tm != NULL );
            ct_tm->abort();
         }
      }
   }
   conflicting_cores.reset(commit_core_id); 

   g_tm_global_statistics.m_num_thread_conflict_per_commit.add2bin(n_conflicting_threads); 
   if (conflicting_cores.any()) 
      g_tm_global_statistics.m_num_core_conflict_per_commit.add2bin(conflicting_cores.count()); 
}

bool ring_tm_manager::committer_abortee_conflict_resolution()
{
   assert(m_ring_starttime >= 0); 
   int conflicting_record = g_tm_ring_global.check_conflict(m_read_word_version, m_ring_starttime); 
   if (conflicting_record > m_ring_starttime) {
      // m_ring_starttime = g_tm_ring_global.ring_index(); // grab a new ring index 
      m_ring_starttime = -1; // just reset the start time 
      // abort tx (by returning true) as it is conflicting with previously committed tx 
      return true; 
   } else {
      g_tm_ring_global.commit_tx(m_thread_uid, m_write_word_set); 
      return false; 
   }
}

// detect + resolve any conflicts
// return true if self-aborted 
bool tm_manager::at_commit_validation()
{
   g_tm_global_state.estimate_coherence_traffic(m_write_set, m_thread_uid); 

   bool self_abort = false; 
	if( !m_timing_mode ) {
      // abort all conflicting transactions
      committer_win_conflict_resolution(); 
	}
   return self_abort; 
}

void tm_manager::at_commit_success()
{
	if( !m_timing_mode ) {
		// remove state from conflict lookup tables
		g_tm_global_state.remove_tuid(m_thread_uid,m_read_set,m_write_set);
		g_tm_global_state.unregister_thread(m_thread);
	}
}

bool tm_manager::commit(bool auto_self_abort)
{
	// Used in NESTED transactions !!!!!!!!
	/*
	printf("[TommyTxn] tm_manager::commit called at cycle %llu, block(%d,%d,%d), thread(%d,%d,%d)\n",
		gpu_sim_cycle + gpu_tot_sim_cycle,
		m_thread->m_ctaid.x, m_thread->m_ctaid.y, m_thread->m_ctaid.z,
		m_thread->m_tid.x,   m_thread->m_tid.y,   m_thread->m_tid.z
	);*/
	m_nesting_level--;
	if( m_nesting_level > 0 ) {
		return false; // flattened closed nesting
	}
	// assume if we did not abort by the time we execute tcommit then we can commit
#ifdef DEBUG_TM
	if (watched()) {
		dim3 tid = m_thread->get_tid();
		dim3 ctaid = m_thread->get_ctaid();
		printf( "[%Lu] begin committing transaction %u (uid=%u) for tid=(%u,%u,%u) cta=(%u,%u,%u) tuid=%u, sc=%u, hwtid=%u\n",
				gpu_sim_cycle,
				m_thread->tm_num_transactions(), m_uid,
				tid.x, tid.y, tid.z, ctaid.x, ctaid.y, ctaid.z,
				m_thread_uid, m_thread_sc, m_thread_hwtid );
	}
#endif

   // step 1: detect + resolve any conflicts 
   bool self_abort = at_commit_validation(); 
   if (self_abort) {
      if (auto_self_abort) abort(); 
      return false; // tell ptx_thread to not delete this manager yet 
   }

   // at this point, the transaction will always commit 
	m_stats.m_abort=0;
	write_stats();

	// step 2: commit updates to memory
   bool writing_tx = false; 
	std::list<access_record>::iterator i;
	for( i=m_write_data.begin(); i != m_write_data.end(); i++ ) {
		access_record &w = *i;
		w.commit();
      writing_tx = true; 
	}

   m_thread->tm_commit();
   g_tm_global_statistics.m_regs_buffered_max.add2bin(m_thread->m_tm_regs_buffered_max);
   g_tm_global_statistics.m_regs_modified_max.add2bin(m_thread->m_tm_regs_modified_max);
   g_tm_global_statistics.m_regs_read_max.add2bin(m_thread->m_tm_regs_read_max);

   // step 3: update global state if commit is successful 
   at_commit_success(); 

	m_thread->end_transaction();

   // update statistics 
	g_tm_global_statistics.m_n_commits += 1;
   if (writing_tx) g_tm_global_statistics.m_n_writing_commits += 1;
	g_tm_global_statistics.dec_concurrency();
   g_tm_global_statistics.record_commit_tx_size(m_read_word_set.size(), m_write_word_set.size(), m_access_word_set.size());
   g_tm_global_statistics.record_tx_blockcount(m_read_block_set, m_write_block_set, m_access_block_set); 
   g_tm_global_statistics.record_raw_info(m_raw_set.size(), m_raw_access); 
   g_tm_global_statistics.m_aborts_per_transaction.add2bin(m_abort_count); 
   g_tm_global_statistics.m_duration.add2bin(gpu_sim_cycle - m_start_cycle); 
   g_tm_global_statistics.m_duration_first_rd.add2bin(gpu_sim_cycle - m_first_read_cycle); 
   g_tm_global_statistics.m_write_buffer_footprint.add2bin(m_buffered_write_word_set.size()); 
   g_tm_global_statistics.m_n_read.add2bin(m_n_read); 
   g_tm_global_statistics.m_n_write.add2bin(m_n_write); 
   g_tm_global_statistics.m_n_rewrite.add2bin(m_n_rewrite / g_tm_options.m_word_size); 
   write_access_log(); 

#ifdef DEBUG_TM
	printf("[%Lu] finished committing transaction %u (uid=%u) for tid=(%u,%u,%u) cta=(%u,%u,%u) tuid=%u, sc=%u, hwtid=%u\n", 
		   gpu_sim_cycle,
		   m_thread->tm_num_transactions(), m_uid,
		   tid.x, tid.y, tid.z, ctaid.x, ctaid.y, ctaid.z, 
		   m_thread_uid, m_thread_sc, m_thread_hwtid );
#else
   if (g_tm_global_statistics.m_n_commits % TM_MSG_INV == 0) {
      g_tm_global_statistics.print_short(stdout); 
   }
#endif
	return true;
}

void tm_manager::accessmode( int readmode, int writemode )
{
   switch (readmode) {
      case 0: m_read_conflict_detection = false; break; 
      case 1: m_read_conflict_detection = true; break; 
      default: assert(0);
   }; 

   switch (writemode) {
      case 0: m_version_management = false; m_write_conflict_detection = false; break; 
      case 1: m_version_management = false; m_write_conflict_detection = true;  break; 
      case 2: m_version_management = true;  m_write_conflict_detection = false; break; 
      case 3: m_version_management = true;  m_write_conflict_detection = true;  break; 
      default: assert(0);
   };

   assert(m_version_management == true); 
}

// the default tm manager does not do commit-time validation, so it cannot uses these 
bool tm_manager::validate_addr( addr_t addr ) { assert(0); return false; }
void tm_manager::commit_addr( addr_t addr ) { assert(0); }
void tm_manager::commit_core_side( ) { assert(0); }
void tm_manager::validate_or_crash( ) { assert(0); }

void tm_manager::print_tm_mem(FILE *fp){

	fprintf(fp,"\nTM MEMORY ACCESSES ");
	fprintf(fp,"\n Address:Contents\n");
	std::list<access_record>::iterator i;
		for( i=m_write_data.begin(); i != m_write_data.end(); i++ ) {
			access_record &w = *i;
			// fprintf(fp,"\n%x:%x",w.getaddr(),*((int*)(w.getvalue())));
         w.print(fp); 
         fprintf(fp,"\n"); 
		}

}

void tm_manager::print_read_write_set(FILE *fp)
{
   fprintf(fp, "TM READ SET:\n"); 
   for (addr_set_t::iterator r = m_read_set.begin(); r != m_read_set.end(); ++r) {
      fprintf(fp, "%x\n", *r); 
   }
   fprintf(fp, "TM WRITE SET:\n"); 
   for (addr_set_t::iterator w = m_write_set.begin(); w != m_write_set.end(); ++w) {
      fprintf(fp, "%x\n", *w); 
   }
}

void tm_manager::write_stats(){
	if (1) {
		m_stats.m_cache_lines_accessed.clear();
		return; 
	}

	// Either the transaction commits or aborts
	// If it commits ,the stats are written to file and the tm_manager is destroyed
	// If it aborts, the stats are written to file and reset
	FILE *fp = fopen("transaction_stats.csv","a");
    assert(fp);

	fprintf(fp,"%zd,",m_stats.m_cache_lines_accessed.size());
	m_stats.m_cache_lines_accessed.clear();

	fprintf(fp,"%ld,",m_read_set.size());

	fprintf(fp,"%ld,",m_write_set.size());

	if(m_write_set.size()>4)
		m_stats.m_overflow=1;
	fprintf(fp,"%d,",m_stats.m_overflow);
		m_stats.m_overflow=0;
	fprintf(fp,"%d\n",m_stats.m_abort);
	//fprintf(fp,"%d\n",m_abort_count);
	if(m_stats.m_abort==1)
		m_stats.m_abort=0;
	
	fclose(fp);
}

void tm_manager::access_record::print(FILE *fout) 
{
   fprintf(fout, "%c", ((m_rd)? 'R':'W'));
   fprintf(fout, "[%#08x]", m_addr);
   unsigned int datab[4] = { 0,0,0,0 }; // 128-bit is the max for a instruction 
   memcpy(datab, m_bytes, m_nbytes); 
   fprintf(fout, "("); 
   for (int i = 0; i < (m_nbytes/4); i++) {
      fprintf(fout, "%x", datab[i]);
      if (i + 1 < (m_nbytes/4)) fprintf(fout, " "); 
   }
   // fprintf(fout, ")@<M=%llu>, ", m_mem); 
   fprintf(fout, "), "); 
}

void tm_manager::write_access_log()
{
   if (g_tm_options.m_access_log_name == NULL) return; 

   FILE *fp = fopen(g_tm_options.m_access_log_name,"a");
   assert(fp);
   fprintf(fp, "[xid=%4d,tuid=%5d]@%llu-%llu: ", m_uid, m_thread_uid, m_start_cycle, gpu_sim_cycle);
   for (std::list<access_record>::iterator iAcc = m_access_log.begin(); iAcc != m_access_log.end(); ++iAcc) {
      iAcc->print(fp); 
   }
   fprintf(fp, "\n");
   fclose(fp);
}

ring_tm_manager::ring_tm_manager( ptx_thread_info *thread, bool timing_mode )
   : tm_manager(thread, timing_mode), m_ring_starttime(-1)
{
   assert(g_tm_options.m_use_ring_tm); 
}

ring_tm_manager::~ring_tm_manager()
{ }

void ring_tm_manager::at_start()
{
   // m_ring_starttime = g_tm_ring_global.ring_index(); 
   m_ring_starttime = -1; 
   // printf("RingTM: txid = %u start\n", m_uid); 
}

// do eager conflict detection: update conflict tx set 
bool ring_tm_manager::at_access( memory_space *mem, bool potential_conflicting, bool rd, addr_t addr, int nbytes )
{
	if( !m_timing_mode and potential_conflicting) {
      // validate the read set so far (the newest read will obtain update data)
      bool self_abort = ring_tm_eager_conflict_resolution(rd); 
      if (self_abort) return true; 

      // obtain version number for current read (if this is a new read)
      ring_tm_access(addr, nbytes, rd); 
	}
   return false; 
}

void ring_tm_manager::at_abort()
{
   m_first_read_cycle = -1; 
   m_read_word_version.clear(); 
}

// detect + resolve any conflicts
// return true if self-aborted 
bool ring_tm_manager::at_commit_validation()
{
   bool self_abort = false; 
   // printf("RingTM: txid = %u commit, version_check(%d %d)\n", m_uid, m_ring_starttime, g_tm_ring_global.ring_index()); 
	if( !m_timing_mode ) {
      self_abort = committer_abortee_conflict_resolution(); 
	}
   return self_abort; 
}

void ring_tm_manager::at_commit_success()
{
   // TODO: Move commit record creation here 
}

// not implemented for ring tm manager for now 
bool ring_tm_manager::validate_addr( addr_t addr ) { assert(0); return false; }
void ring_tm_manager::commit_addr( addr_t addr ) { assert(0); }
void ring_tm_manager::commit_core_side( ) { assert(0); }

void ring_tm_manager::print(FILE *fout)
{
   addr_version_set_t::const_iterator iAddrVer; 
   for (iAddrVer = m_read_word_version.begin(); iAddrVer != m_read_word_version.end(); ++iAddrVer) {
      fprintf(fout, "[%#08x] v%d\n", iAddrVer->first, iAddrVer->second); 
   }
}


bool conflict_set::conflict() const 
{
   size_t n_writers = m_tuids_have_written.size(); 
   size_t n_readers = m_tuids_have_read.size(); 
   if (n_writers > 1) {
      return true; 
   } else if (n_writers == 1 and n_readers > 0) {
      return true; 
   } else {
      return false; 
   }
}

void tm_sample_conflict_footprint()
{
   g_tm_global_statistics.sample_footprint(); 
}

void tm_global_statistics::sample_footprint()
{
   // sample the maximum within the sampling interval 
   m_conflict_footprint.add2bin(m_max_conflict_footprint); 
   m_max_conflict_footprint = 0; 

   m_transaction_footprint.add2bin(m_max_transaction_footprint); 
   m_max_transaction_footprint = 0; 

   m_readonly_footprint.add2bin(m_max_readonly_footprint); 
   m_max_readonly_footprint = 0; 

   m_writeonly_footprint.add2bin(m_max_writeonly_footprint); 
   m_max_writeonly_footprint = 0; 
}

template<class T> 
void retainMax(T& max_value, T test_value)
{
   max_value = std::max(max_value, test_value); 
}

void track_delta_change(bool before, bool after, int& value)
{
   if (before and not after) { // 0 --> 1
      value -= 1; 
   } else if (not before and after) { // 1 --> 0
      value += 1; 
   }
}

tuid_set_t tm_global_state::mem_access( unsigned tuid, bool rd, unsigned addr, unsigned nbytes )
{
   // piggy-bagging bloomfilter update here 
   memaccess_tx_bf(tuid, rd, addr, nbytes); 

	tuid_set_t result;
	for( unsigned a=addr; a < (addr+nbytes); ++a ) {
		conflict_set &cset = m_mem[a];
      bool conflict_state_before = cset.conflict(); 
      bool read_only_before = cset.read_only(); 
      bool write_only_before = cset.write_only(); 
		if( rd ) {
			cset.m_tuids_have_read.insert(tuid);
			tuid_set_t::iterator written_begin = cset.m_tuids_have_written.begin() ;
			tuid_set_t::iterator written_end = cset.m_tuids_have_written.end() ;
//			//result.insert( cset.m_tuids_have_written.begin(), cset.m_tuids_have_written.end() );
         if(!cset.m_tuids_have_written.empty())
            result.insert( written_begin,written_end );
		} else {
			cset.m_tuids_have_written.insert(tuid);
			tuid_set_t::iterator written_begin = cset.m_tuids_have_written.begin() ;
			tuid_set_t::iterator written_end = cset.m_tuids_have_written.end() ;

			tuid_set_t::iterator read_begin = cset.m_tuids_have_read.begin() ;
			tuid_set_t::iterator read_end = cset.m_tuids_have_read.end() ;

			if(!cset.m_tuids_have_read.empty())
				result.insert( read_begin,read_end );
			if(!cset.m_tuids_have_written.empty())
				result.insert( written_begin,written_end );
		}
      bool conflict_state_after = cset.conflict(); 
      bool read_only_after = cset.read_only(); 
      bool write_only_after = cset.write_only(); 

      // update conflict footprint if there is a change in conflict state 
      track_delta_change(conflict_state_before, conflict_state_after, m_conflict_footprint); 
      retainMax<int>(g_tm_global_statistics.m_max_conflict_footprint, m_conflict_footprint); 
      track_delta_change(read_only_before, read_only_after, m_readonly_footprint); 
      retainMax<int>(g_tm_global_statistics.m_max_readonly_footprint, m_readonly_footprint); 
      track_delta_change(write_only_before, write_only_after, m_writeonly_footprint); 
      retainMax<int>(g_tm_global_statistics.m_max_writeonly_footprint, m_writeonly_footprint); 
	}
   retainMax<size_t>(g_tm_global_statistics.m_max_transaction_footprint, m_mem.size()); 
	return result;
}

// estimate the ammount of traffice required if coherence were to be use for TM implementation
void tm_global_state::estimate_coherence_traffic(const addr_set_t& write_set, unsigned commit_tuid)
{
   unsigned coherence_traffic_thread_level = 0; // traffice sent between threads 
   unsigned coherence_traffic_core_level = 0; 
   addr_set_t::const_iterator i_write_addr = write_set.begin(); 
   for (; i_write_addr != write_set.end(); ++i_write_addr) {
      const conflict_set &cset = m_mem[*i_write_addr]; 
      // this byte needs to be sent to every thread that conflicts 
      tuid_set_t conflict_threads(cset.m_tuids_have_read); // read-write conflict 
      conflict_threads.insert(cset.m_tuids_have_written.begin(), cset.m_tuids_have_written.end()); // write-write conflict 
      conflict_threads.erase(commit_tuid); 
      coherence_traffic_thread_level += conflict_threads.size();  

      // calculate #cores to which this byte needs to be sent 
      std::set<unsigned> conflict_core; 
      for (tuid_set_t::iterator ithread = conflict_threads.begin(); 
           ithread != conflict_threads.end(); ++ithread)
      {
         // determine the conflicting core 
         ptx_thread_info *ct = lookup_tuid(*ithread);
         unsigned conflict_core_id = ct->get_hw_sid(); 
         conflict_core.insert(conflict_core_id); 
      }
      ptx_thread_info *commit_thread = lookup_tuid(commit_tuid); 
      conflict_core.erase(commit_thread->get_hw_sid()); 
      coherence_traffic_core_level += conflict_core.size(); 
   }
   g_tm_global_statistics.m_lazy_coherence_traffic_thread_level += coherence_traffic_thread_level;
   g_tm_global_statistics.m_lazy_coherence_traffic_core_level += coherence_traffic_core_level;
   g_tm_global_statistics.m_lazy_coherence_traffic.add2bin(coherence_traffic_core_level); 
}

void tm_global_state::memaccess_tx_bf( unsigned tuid, bool rd, unsigned addr, unsigned nbytes )
{
   // determine the core doing this access
	ptx_thread_info *ct = lookup_tuid(tuid);
	assert( ct != NULL ); 
   unsigned accessor_core_id = ct->get_hw_sid(); 
   
   tm_bloomfilter_set &thread_bfaccess = m_bfaccess[tuid]; 
   thread_bfaccess.set_core_set( accessor_core_id, &(m_bfcore[accessor_core_id]) ); 
   thread_bfaccess.mem_access(rd, addr, nbytes);

   // determine the hw thread inside the core doing this access
   unsigned accessor_hwtid = ct->get_hw_tid(); 

   // update core bloomfilter (hashed version)
   tm_bloomfilter_hashed_core_set &core_bfaccess = m_bfcorehash[accessor_core_id]; 
   core_bfaccess.set_core_id(accessor_core_id); 
   core_bfaccess.mem_access(accessor_hwtid, rd, addr, nbytes); 
}

void tm_global_state::register_thread( ptx_thread_info *thrd )
{
	m_tlookup[ thrd->get_uid() ] = thrd;
   unsigned core_id = thrd->get_hw_sid(); 
   unsigned thread_id = thrd->get_hw_tid(); 
   m_threads_in_tx[core_id].set(thread_id); 
}

void tm_global_state::unregister_thread( ptx_thread_info *thrd )
{
	m_tlookup.erase( thrd->get_uid() );
   unsigned core_id = thrd->get_hw_sid(); 
   unsigned thread_id = thrd->get_hw_tid(); 
   m_threads_in_tx[core_id].reset(thread_id); 
}

ptx_thread_info *tm_global_state::lookup_tuid(unsigned tuid)
{
	tuid_to_thread_t::iterator t = m_tlookup.find(tuid);
	if( t == m_tlookup.end() ) {
		return NULL;
	} else {
		return t->second;
	}
}

void tm_global_state::remove_tuid( unsigned tuid, const addr_set_t &read_set, const addr_set_t &write_set )
{
	for( addr_set_t::iterator r=read_set.begin(); r != read_set.end(); ++r ) {
		unsigned addr = *r;
		conflict_set &cset = m_mem[addr];
      bool conflict_before = cset.conflict(); 
      bool read_only_before = cset.read_only(); 
      bool write_only_before = cset.write_only(); 
		cset.m_tuids_have_read.erase(tuid);
      bool conflict_after = cset.conflict(); 
      bool read_only_after = cset.read_only(); 
      bool write_only_after = cset.write_only(); 
      if (conflict_before and not conflict_after) m_conflict_footprint -= 1; 
      assert(m_conflict_footprint >= 0); 
      track_delta_change(read_only_before, read_only_after, m_readonly_footprint); 
      retainMax<int>(g_tm_global_statistics.m_max_readonly_footprint, m_readonly_footprint); 
      track_delta_change(write_only_before, write_only_after, m_writeonly_footprint); 
      retainMax<int>(g_tm_global_statistics.m_max_writeonly_footprint, m_writeonly_footprint); 
		if( cset.m_tuids_have_read.empty() && cset.m_tuids_have_written.empty() ) 
			m_mem.erase(addr);
	}
	for( addr_set_t::iterator w=write_set.begin(); w != write_set.end(); ++w ) {
		unsigned addr = *w;
		conflict_set &cset = m_mem[addr];
      bool conflict_before = cset.conflict(); 
      bool read_only_before = cset.read_only(); 
      bool write_only_before = cset.write_only(); 
		cset.m_tuids_have_written.erase(tuid);
      bool conflict_after = cset.conflict(); 
      bool read_only_after = cset.read_only(); 
      bool write_only_after = cset.write_only(); 
      if (conflict_before and not conflict_after) m_conflict_footprint -= 1; 
      assert(m_conflict_footprint >= 0); 
      track_delta_change(read_only_before, read_only_after, m_readonly_footprint); 
      retainMax<int>(g_tm_global_statistics.m_max_readonly_footprint, m_readonly_footprint); 
      track_delta_change(write_only_before, write_only_after, m_writeonly_footprint); 
      retainMax<int>(g_tm_global_statistics.m_max_writeonly_footprint, m_writeonly_footprint); 
		if( cset.m_tuids_have_read.empty() && cset.m_tuids_have_written.empty() ) 
			m_mem.erase(addr);
	}
   retainMax<size_t>(g_tm_global_statistics.m_max_transaction_footprint, m_mem.size()); 
}

tuid_set_t tm_global_state::lazy_conflict_detection( const addr_set_t& write_set )
{
   tuid_set_t conflicting_threads; 

   for (addr_set_t::const_iterator w = write_set.begin(); w != write_set.end(); ++w) {
      conflict_set& cs = m_mem[*w]; 
      conflicting_threads.insert(cs.m_tuids_have_read.begin(), cs.m_tuids_have_read.end()); 
      conflicting_threads.insert(cs.m_tuids_have_written.begin(), cs.m_tuids_have_written.end()); 
      g_tm_global_statistics.record_abort_at_address(*w, cs); 
   }

   return conflicting_threads; 
}

void tm_global_statistics::record_abort_at_address(addr_t addr, const conflict_set& cs) 
{
   if (g_tm_options.m_abort_profile == false) return; 

   // construct aborted tx set (can be read/write overlap)
   tuid_set_t aborted(cs.m_tuids_have_read); 
   aborted.insert(cs.m_tuids_have_written.begin(), cs.m_tuids_have_written.end()); 

   // add the size of this set to the abort profile (minus 1 to discount for the committing tx)
   if (aborted.size() > 1) 
      m_abort_profile[addr] += aborted.size() - 1; 
}

void tm_global_statistics::dump_abort_profile(FILE *csv) 
{
   tr1_hash_map<addr_t, unsigned int>::const_iterator iAddr; 
   float threshold = 0.0f; // m_n_aborts / 100.f; 
   for (iAddr = m_abort_profile.begin(); iAddr != m_abort_profile.end(); ++iAddr) {
      if (iAddr->second > threshold) {
         fprintf(csv, "%u, %u\n", iAddr->first, iAddr->second); 
      }
   }
}

void tm_dump_profile()
{
   if (g_tm_options.m_abort_profile) {
      FILE* csv = fopen("abort_profile.csv", "w"); 
      g_tm_global_statistics.dump_abort_profile(csv); 
      fclose(csv); 
   } 

   if (g_tm_options.m_ring_dump_record) {
      FILE* commit_rec_f = fopen("commit_rec.txt", "w"); 
      g_tm_ring_global.print(commit_rec_f); 
      fclose(commit_rec_f); 
   }
}

const tm_bloomfilter_set& tm_global_state::get_BF(unsigned tuid) const
{
   return m_bfaccess.find(tuid)->second; 
}

void tm_global_state::commit_tx_bf( unsigned tuid, const addr_set_t& write_set, const tuid_set_t& true_conflict_set )
{
   std::bitset<32> conflicting_cores; 
   conflicting_cores.reset();

   bloomfilter_access_set_t::iterator i_txbf_commit = m_bfaccess.find(tuid); 
   assert(i_txbf_commit != m_bfaccess.end()); 

   tuid_set_t *true_conflict_set_copy = NULL;
   
   if (g_tm_options.m_check_bloomfilter_correctness) {
      true_conflict_set_copy = new tuid_set_t(true_conflict_set); 
      true_conflict_set_copy->erase(tuid); 
   }

   bloomfilter_access_set_t::iterator i_txbf = m_bfaccess.begin(); 
   for (; i_txbf != m_bfaccess.end(); ++i_txbf) {
      if (i_txbf->first == tuid) continue; // do not count self 

      bool bloom_filter_match = (i_txbf->second.match_access_set(i_txbf_commit->second)); 
      if (bloom_filter_match == true) {
         g_tm_global_statistics.m_n_bloomfilter_detected_conflicts += 1; 
         bool true_conflict = (true_conflict_set.find(i_txbf->first) != true_conflict_set.end());
         if (true_conflict == false) {
            g_tm_global_statistics.m_n_bloomfilter_false_conflicts += 1; 
         }
      }

      bool addr_stream_match = (i_txbf->second.match_access_set(write_set));
      if (addr_stream_match == true) {
         // determine the core in conflict 
			ptx_thread_info *ct = lookup_tuid(i_txbf->first);
			assert( ct != NULL ); 
         unsigned conflict_core_id = ct->get_hw_sid(); 
         conflicting_cores.set(conflict_core_id); 

         g_tm_global_statistics.m_n_bfxaddrstrm_detected_conflicts += 1; 
         bool true_conflict = (true_conflict_set.find(i_txbf->first) != true_conflict_set.end());
         if (true_conflict == false) {
            g_tm_global_statistics.m_n_bfxaddrstrm_false_conflicts += 1; 
            tm_manager * conflict_tx = dynamic_cast<tm_manager*>( ct->get_tm_manager() ); 
            g_tm_global_statistics.record_false_conflict_info(write_set.size(), conflict_tx->get_access_size()); 
         }

         if (g_tm_options.m_check_bloomfilter_correctness) {
            true_conflict_set_copy->erase(i_txbf->first); 
         }
      }
   }

   if (g_tm_options.m_check_bloomfilter_correctness) {
      if (true_conflict_set_copy->empty() == false) {
         dynamic_cast<tm_manager*>(lookup_tuid(tuid)->get_tm_manager())->print_read_write_set(stdout); 
         printf("Undetected conflict: "); 
         for (tuid_set_t::iterator i = true_conflict_set_copy->begin(); i != true_conflict_set_copy->end(); ++i) {
            printf("%d ", *i); 
            dynamic_cast<tm_manager*>(lookup_tuid(*i)->get_tm_manager())->print_read_write_set(stdout); 
         }
         printf("\n"); 
         assert(0); 
         }
      delete true_conflict_set_copy; 
      true_conflict_set_copy = NULL; 
   }

   // determine the core committing, remove that core from the conflicting set 
	ptx_thread_info *commit_thread = lookup_tuid(i_txbf_commit->first);
	assert( commit_thread != NULL ); 
   unsigned commit_core_id = commit_thread->get_hw_sid(); 
   unsigned commit_hwtid = commit_thread->get_hw_tid(); 
   conflicting_cores.reset(commit_core_id); 

   if (conflicting_cores.any()) 
      g_tm_global_statistics.m_num_core_bfxaddrstrm_per_commit.add2bin(conflicting_cores.count()); 

   // check the per-core counting bloom filter 
   { 
   std::bitset<32> conflicting_cores_bf; 
   conflicting_cores_bf.reset();

   for (core_bloomfilter_t::iterator i_corebf = m_bfcore.begin(); i_corebf != m_bfcore.end(); ++i_corebf) {
      bool addr_stream_match = (i_corebf->second.match_access_set(write_set));
      unsigned bf_core_id = i_corebf->first; 
      // the core commiting do not need to check this bloomfilter 
      if (addr_stream_match == true and bf_core_id != commit_core_id) {
         conflicting_cores_bf.set(bf_core_id); 
      }
   }

   if (conflicting_cores_bf.any()) {
      g_tm_global_statistics.m_num_core_bfsum_per_commit.add2bin(conflicting_cores_bf.count()); 
      std::bitset<32> conflict_missed = ~conflicting_cores_bf & conflicting_cores; 
      assert( conflict_missed.none() ); 

      std::bitset<32> conflict_extra = conflicting_cores_bf & ~conflicting_cores; 
      g_tm_global_statistics.m_num_core_bfsum_extra_per_commit.add2bin(conflict_extra.count()); 
   }
   }

   // check the per-core thread-hashed bloom filter 
   {
   // create true conflict thread vector for each core 
   std::map<unsigned, hashtable_bits_mt::tvec_t> true_conflict_tvec; 
   for (tuid_set_t::const_iterator it = true_conflict_set.begin(); it != true_conflict_set.end(); ++it) {
      ptx_thread_info *conflict_thread = lookup_tuid(*it);
      unsigned core_id = conflict_thread->get_hw_sid(); 
      unsigned thread_id = conflict_thread->get_hw_tid(); 
      true_conflict_tvec[core_id].set(thread_id); 
   }
   true_conflict_tvec[commit_core_id].reset(commit_hwtid); 
   std::bitset<32> conflicting_cores_bf_hashed; 
   conflicting_cores_bf_hashed.reset();

   for (core_hash_bloomfilter_t::iterator i_corebf = m_bfcorehash.begin(); i_corebf != m_bfcorehash.end(); ++i_corebf) {
      hashtable_bits_mt::tvec_t matched_threads; 
      unsigned int addr_stream_match = i_corebf->second.match_access_set(write_set, matched_threads);
      unsigned bf_core_id = i_corebf->first; 
      if (addr_stream_match > 0) {
         // conflict between core detected 
         conflicting_cores_bf_hashed.set(bf_core_id); 
         // filter the threads that not in transaction 
         matched_threads &= m_threads_in_tx[bf_core_id]; 
         // filter the committing thread itself 
         if (bf_core_id == commit_core_id) matched_threads.reset(commit_hwtid); 
         addr_stream_match = matched_threads.count(); 
         g_tm_global_statistics.m_num_thread_match_per_bfhash_match.add2bin(addr_stream_match); 
         g_tm_global_statistics.m_n_bfxaddrstrm_hashed_detected_conflicts += addr_stream_match; 
         // detect missing conflicts 
         hashtable_bits_mt::tvec_t missing_conflicts = ~matched_threads & true_conflict_tvec[bf_core_id]; 
         assert(missing_conflicts.any() == false); 
         // detect false conflicts 
         matched_threads &= ~(true_conflict_tvec[bf_core_id]); 
         unsigned int false_conflict_threads = matched_threads.count(); 
         g_tm_global_statistics.m_n_bfxaddrstrm_hashed_false_conflicts += false_conflict_threads; 
      }
   }
   conflicting_cores_bf_hashed.reset(commit_core_id); // discounting the committing core itself 

   if (conflicting_cores_bf_hashed.any()) {
      g_tm_global_statistics.m_num_core_bfhash_per_commit.add2bin(conflicting_cores_bf_hashed.count()); 
      std::bitset<32> conflict_missed = ~conflicting_cores_bf_hashed & conflicting_cores; 
      assert( conflict_missed.none() ); 
   }
   }

   // clear the bloomfilter of the committing transaction 
   i_txbf_commit->second.clear(); 
   m_bfcorehash[commit_core_id].clear(commit_hwtid); 
   m_bfaccess.erase(i_txbf_commit); 
}

void tm_global_state::abort_tx_bf( unsigned tuid )
{
   // clear the bloomfilter of the aborting transaction 
   m_bfaccess.find(tuid)->second.clear(); 

	ptx_thread_info *ct = lookup_tuid(tuid);
	assert( ct != NULL ); 
   unsigned abort_core_id = ct->get_hw_sid(); 
   unsigned abort_hwtid = ct->get_hw_tid(); 
   m_bfcorehash[abort_core_id].clear(abort_hwtid); 
}

void tm_global_state::print_resource_usage( FILE *fout ) 
{
   fprintf(fout, "m_mem.size = %zd\n", m_mem.size()); 
   fprintf(fout, "m_tlookup.size = %zd\n", m_tlookup.size()); 
   fprintf(fout, "m_bfaccess.size = %zd\n", m_bfaccess.size()); 
   fprintf(fout, "m_bfcore.size = %zd\n", m_bfcore.size()); 
}

tm_manager::access_record::access_record( const access_record &another )
{
	m_parent = another.m_parent;
	m_mem = another.m_mem;
	m_rd = another.m_rd;
	m_addr = another.m_addr;
	m_nbytes = another.m_nbytes;
	m_bytes = malloc(m_nbytes);
	memcpy(m_bytes,another.m_bytes,m_nbytes);
}

tm_manager::access_record::access_record( tm_manager *parent, memory_space *mem, bool rd, addr_t addr, void *data, unsigned nbytes )
{
	m_parent = parent;
	m_mem = mem;
	m_rd = rd;
	m_addr = addr;
	m_nbytes = nbytes;
	m_bytes = malloc(nbytes);
	memcpy(m_bytes,data,nbytes);
}


tm_manager::access_record::~access_record()
{
	free(m_bytes);
} 

void tm_manager::access_record::commit()
{
	if ( !m_rd ) {
#ifdef DEBUG_TM
		if ( m_nbytes == 4 ) {
			printf("[%Lu]   -> transaction (uid=%u) writing %d to address 0x%08x in mem %p\n", 
				   gpu_sim_cycle,
				   m_parent->uid(), *(int*)m_bytes, m_addr, m_mem ); 
		}
#endif
        m_mem->write(m_addr, m_nbytes, m_bytes, NULL /*thread*/, NULL /*pI*/);
	}
}

// used by value_based_tm_manager for timing driven commits 
void tm_manager::access_record::commit_word( addr_t addr, unsigned nbytes )
{
	assert ( !m_rd ); 

   assert( addr >= m_addr ); 
   assert( m_addr + m_nbytes >= addr + nbytes ); 

   addr_t offset = addr - m_addr; 
   char * data = ((char *) m_bytes) + offset;
   ptx_thread_info* thd = NULL; ptx_instruction* pI = NULL;
   thd = this->m_parent->m_thread;
   if (thd) pI = (ptx_instruction*)(thd->get_inst());
   m_mem->write(addr, nbytes, (void*)data, thd, pI);
}

// return true if the access_record contain the given address range [addr, addr + nbytes)
bool tm_manager::access_record::contain_addr( addr_t addr, unsigned nbytes )
{
   addr_t access_bound[2] = { m_addr, m_addr + m_nbytes };
   addr_t word_bound[2] = { addr, addr + nbytes }; 

   bool in_bound = (access_bound[0] <= word_bound[0]) and (access_bound[1] >= word_bound[1]); 

   return in_bound; 
}

void tm_global_statistics::record_tx_blockcount(const addr_set_t& read_set_block, 
                                                const addr_set_t& write_set_block, 
                                                const addr_set_t& access_set_block) 
{
    size_t read_size = read_set_block.size(); 
    size_t write_size = write_set_block.size();
    size_t access_size = access_set_block.size();

    m_read_nblock.add2bin(read_size); 
    m_write_nblock.add2bin(write_size); 
    m_total_nblock.add2bin(access_size); 

    // count the number of memory bank touched by this transactions 
    std::bitset<32> mem_bank_touched; 
    mem_bank_touched.reset(); 

    for (addr_set_t::iterator iblock = access_set_block.begin(); iblock != access_set_block.end(); ++iblock) {
        unsigned bank_id = *(iblock) % g_tm_options.m_nbank; 
        mem_bank_touched.set(bank_id); 
    }
    m_total_nmem.add2bin(mem_bank_touched.count()); 
}

void tm_global_statistics::print(FILE *fout) 
{
    fprintf(fout, "tm_n_aborts = %llu \n", m_n_aborts); 
    fprintf(fout, "tm_n_commits = %llu \n", m_n_commits); 
    fprintf(fout, "tm_n_writing_commits = %llu \n", m_n_writing_commits); 
    fprintf(fout, "tm_n_transactions = %llu \n", m_n_transactions); 
    fprintf(fout, "tm_n_intra_warp_detected_conflicts = %llu \n", m_n_intra_warp_detected_conflicts); 
    fprintf(fout, "tm_n_vcd_tcd_mismatch = %llu \n", m_n_vcd_tcd_mismatch); 
    m_aborts_per_transaction.fprint(fout); fprintf(fout, "\n"); 
    m_duration.fprint(fout); fprintf(fout, "\n"); 
    m_duration_first_rd.fprint(fout); fprintf(fout, "\n"); 
    fprintf(fout, "tm_max_concurrency = %u \n", m_max_concurrency); 
    fprintf(fout, "tm_n_rollback_insn = %llu \n", m_n_rollback_insn); 
    m_n_insn_per_aborted_txn.fprint(fout); fprintf(fout, "\n");
    m_n_insn_per_committed_txn.fprint(fout); fprintf(fout, "\n");
    fprintf(fout, "tm_n_insn_txn = %llu \n", m_n_insn_txn);

    fprintf(fout, "tm_n_intra_warp_aborts  = %llu \n", m_n_intra_warp_aborts ); 
    fprintf(fout, "tm_n_intra_block_aborts = %llu \n", m_n_intra_block_aborts); 
    fprintf(fout, "tm_n_intra_core_aborts  = %llu \n", m_n_intra_core_aborts ); 

    fprintf(fout, "tm_bloomfilter_detected_conflicts = %llu \n", m_n_bloomfilter_detected_conflicts); 
    fprintf(fout, "tm_bloomfilter_false_conflicts = %llu \n", m_n_bloomfilter_false_conflicts); 

    fprintf(fout, "tm_bfxaddrstrm_detected_conflicts = %llu \n", m_n_bfxaddrstrm_detected_conflicts); 
    fprintf(fout, "tm_bfxaddrstrm_false_conflicts = %llu \n", m_n_bfxaddrstrm_false_conflicts); 

    fprintf(fout, "tm_bfxaddrstrm_hashed_detected_conflicts = %llu \n", m_n_bfxaddrstrm_hashed_detected_conflicts); 
    fprintf(fout, "tm_bfxaddrstrm_hashed_false_conflicts = %llu \n", m_n_bfxaddrstrm_hashed_false_conflicts); 

    m_read_size.fprint(fout); fprintf(fout, "\n"); 
    m_write_size.fprint(fout); fprintf(fout, "\n"); 
    m_total_size.fprint(fout); fprintf(fout, "\n"); 

    m_read_sz_all.fprint(fout); fprintf(fout, "\n"); 
    m_write_sz_all.fprint(fout); fprintf(fout, "\n"); 
    m_total_sz_all.fprint(fout); fprintf(fout, "\n"); 

    m_read_nblock.fprint(fout); fprintf(fout, "\n"); 
    m_write_nblock.fprint(fout); fprintf(fout, "\n"); 
    m_total_nblock.fprint(fout); fprintf(fout, "\n"); 
    m_total_nmem.fprint(fout); fprintf(fout, "\n"); 

    m_num_thread_conflict_per_commit.fprint(fout); fprintf(fout, "\n"); 
    m_num_core_conflict_per_commit.fprint(fout); fprintf(fout, "\n"); 
    m_num_core_bfxaddrstrm_per_commit.fprint(fout); fprintf(fout, "\n"); 

    m_num_core_bfsum_per_commit.fprint(fout); fprintf(fout, "\n"); 
    m_num_core_bfsum_extra_per_commit.fprint(fout); fprintf(fout, "\n"); 

    m_num_core_bfhash_per_commit.fprint(fout); fprintf(fout, "\n"); 

    fprintf(fout, "tm_lazy_coherence_traffic_thread_level = %llu \n", m_lazy_coherence_traffic_thread_level); 
    fprintf(fout, "tm_lazy_coherence_traffic_core_level = %llu \n", m_lazy_coherence_traffic_core_level); 
    m_lazy_coherence_traffic.fprint(fout); fprintf(fout, "\n"); 

    m_false_aborter_size.fprint(fout); fprintf(fout, "\n"); 
    m_false_abortee_size.fprint(fout); fprintf(fout, "\n"); 

    m_raw_footprint.fprint(fout); fprintf(fout, "\n"); 
    m_raw_access.fprint(fout); fprintf(fout, "\n"); 

    m_conflict_footprint.fprint(fout); fprintf(fout, "\n"); 
    m_transaction_footprint.fprint(fout); fprintf(fout, "\n"); 
    m_readonly_footprint.fprint(fout); fprintf(fout, "\n"); 
    m_writeonly_footprint.fprint(fout); fprintf(fout, "\n"); 

    m_num_thread_match_per_bfhash_match.fprint(fout); fprintf(fout, "\n"); 

    m_n_reread.fprint(fout); fprintf(fout, "\n");
    fprintf(fout, "tm_n_reread_violation = %u\n", m_n_reread_violation); 

    m_write_buffer_footprint.fprint(fout); fprintf(fout, "\n"); 

    m_n_rewrite.fprint(fout); fprintf(fout, "\n"); 
    m_n_read.fprint(fout); fprintf(fout, "\n"); 
    m_n_write.fprint(fout); fprintf(fout, "\n"); 

    m_n_timeout_validation.fprint(fout); fprintf(fout, "\n"); 

    m_n_warp_level_raw.fprint(fout); fprintf(fout, "\n"); 

    m_regs_modified_max.fprint(fout); fprintf(fout, "\n");
    m_regs_buffered_max.fprint(fout); fprintf(fout, "\n");
    m_regs_read_max.fprint(fout); fprintf(fout, "\n");
}

void tm_global_statistics::print_short(FILE *fout) 
{
    fprintf(fout, "[%Lu] ", gpu_sim_cycle); 
    fprintf(fout, "tm_n_transactions = %6llu tm_n_commits = %6llu tm_n_aborts = %6llu tm_n_writing_commits = %6llu\n", 
            m_n_transactions, m_n_commits, m_n_aborts, m_n_writing_commits); 
}

void tm_global_statistics::visualizer_print(gzFile visualizer_file) 
{
    gzprintf(visualizer_file, "tm_n_transactions: %llu\n", m_n_transactions); 
    gzprintf(visualizer_file, "tm_n_writing_commits: %llu\n", m_n_writing_commits); 
    gzprintf(visualizer_file, "tm_n_commits: %llu\n", m_n_commits); 
    gzprintf(visualizer_file, "tm_n_aborts: %llu\n", m_n_aborts); 
}

void tm_statistics_visualizer( gzFile visualizer_file ) 
{
    g_tm_global_statistics.visualizer_print(visualizer_file); 
}

void tm_statistics(FILE *fout)
{
    g_tm_global_statistics.print(fout); 
    g_tm_ring_stats.print(fout); 
}

void tm_reg_options(option_parser_t opp) 
{
   g_tm_options.reg_options(opp); 
   g_tm_bloomfilter_options.reg_options(opp); 
}

tm_options::tm_options()
   : m_derive_done(false), m_word_size(4), m_word_size_log2(2), 
     m_access_block_size(256), m_access_block_size_log2(8), 
     m_abort_profile(false), 
     m_use_ring_tm(false), m_ring_tm_eager_cd(false)
{ }

void tm_options::check_n_derive()
{
   if (m_derive_done) return; 

   m_word_size_log2 = 0; 
   while ((1 << m_word_size_log2) < m_word_size) {
      m_word_size_log2 += 1;
   }

   m_access_block_size_log2 = 0; 
   while ((1 << m_access_block_size_log2) < m_access_block_size) {
      m_access_block_size_log2 += 1;
   }

   if (m_use_ring_tm) assert(!m_use_value_based_tm); 
   if (m_use_value_based_tm) assert(!m_use_ring_tm); 

   if (m_compressed_ring_capacity != 0) 
      assert(m_compressed_ring_size != 0); 

   if (m_abort_profile or m_ring_dump_record) 
      atexit(tm_dump_profile); 

   if (m_timing_mode_core_side_commit) {
      bool timing_mode_vb_commit; 
      option_parser_getvalue(m_opp, "-timing_mode_vb_commit", OPT_BOOL, &timing_mode_vb_commit); 
      assert(timing_mode_vb_commit == true); 
   }

   m_temporal_cd_addr_granularity_log2 = 0; 
   while ((1 << m_temporal_cd_addr_granularity_log2) < m_temporal_cd_addr_granularity) {
      m_temporal_cd_addr_granularity_log2 += 1; 
   }

   m_derive_done = true; 
}

void tm_options::reg_options(option_parser_t opp)
{
   m_opp = opp; // retaining a handle of the option parser 

   option_parser_register(opp, "-tm_access_word_size", OPT_UINT32, &m_word_size, 
               "size of the word for tm access tracking in bloom filters (default = 4)",
               "4");
   option_parser_register(opp, "-tm_access_block_size", OPT_UINT32, &m_access_block_size, 
               "size of the block for tm access tracking (default = 256)",
               "256");
   option_parser_register(opp, "-tm_conflict_table_bank", OPT_UINT32, &m_nbank, 
               "number of banks for the tm access tracking table (default = 8)",
               "8");
   option_parser_register(opp, "-tm_lazy_conflict_detection", OPT_BOOL, &m_lazy_conflict_detection, 
               "use lazy conflict detection in TM (default = off)",
               "0");
   option_parser_register(opp, "-tm_check_bloomfilter_correctness", OPT_BOOL, &m_check_bloomfilter_correctness, 
               "check for false-negatives in bloomfilter-based conflict detection (default = off)",
               "0");
   option_parser_register(opp, "-tm_abort_profile", OPT_BOOL, &m_abort_profile, 
               "create a profile of abort causing memory locations (default = off)",
               "0");
   option_parser_register(opp, "-tm_use_ring_tm", OPT_BOOL, &m_use_ring_tm, 
               "use ring TM mechanism (committer-abortee) (default = off)",
               "0");
   option_parser_register(opp, "-tm_ring_tm_eager_cd", OPT_BOOL, &m_ring_tm_eager_cd, 
               "use ring TM with eager conflict detection (default = off)",
               "0");
   option_parser_register(opp, "-tm_ring_tm_size_limit", OPT_INT32, &m_ring_tm_size_limit, 
               "number of entries in the ring TM (default = 0 = unlimited)",
               "0");
   option_parser_register(opp, "-tm_ring_tm_bloomfilter", OPT_BOOL, &m_ring_tm_bloomfilter, 
               "use bloomfilter in ringTM commit record (default = off)",
               "0");
   option_parser_register(opp, "-tm_ring_tm_record_capacity", OPT_INT32, &m_ring_tm_record_capacity, 
               "maximum number of words in each ringTM commit record (default = 0 = unlimited)",
               "0");
   option_parser_register(opp, "-tm_ring_tm_version_read", OPT_BOOL, &m_ring_tm_version_read, 
               "tag a version to reach member in read set (default = off)",
               "0");
   option_parser_register(opp, "-tm_compressed_ring_capacity", OPT_INT32, &m_compressed_ring_capacity, 
               "maximum number of commit record in each compressed ring record (default = 0 = not used)",
               "0");
   option_parser_register(opp, "-tm_compressed_ring_size", OPT_INT32, &m_compressed_ring_size, 
               "maximum number of true size of each compressed ring record (default = 0 = not used)",
               "0");
   option_parser_register(opp, "-tm_ring_dump_record", OPT_BOOL, &m_ring_dump_record, 
               "create a dump of all commit record (default = off)",
               "0");
   option_parser_register(opp, "-tm_use_value_based_tm", OPT_BOOL, &m_use_value_based_tm, 
               "use value-based conflict detection TM mechanism (committer-abortee) (default = off)",
               "0");
   option_parser_register(opp, "-tm_value_based_eager_cr", OPT_BOOL, &m_value_based_eager_cr, 
               "use value-based eager conflict resolution (default = off)",
               "0");
   option_parser_register(opp, "-tm_enable_access_mode", OPT_BOOL, &m_enable_access_mode, 
               "recognize intrinsics that specify if a access can potentially be conflicting (default = off)",
               "0");
   option_parser_register(opp, "-tm_timeout_validation", OPT_INT32, &m_timeout_validation, 
               "at timeout, perform a validation pass (and self-abort if violated) (default = 0 = no timeout)",
               "0");
   option_parser_register(opp, "-tm_access_log_name", OPT_CSTR, &m_access_log_name, 
               "file name of the transactional memory access log (default = NULL = no log)", 
               NULL); 
   option_parser_register(opp, "-timing_mode_core_side_commit", OPT_BOOL, &m_timing_mode_core_side_commit, 
               "commit everything at core-side commit (default = off)",
               "0");
   option_parser_register(opp, "-tm_exact_temporal_conflict_detection", OPT_BOOL, &m_exact_temporal_conflict_detection, 
               "Unlimited storage for timestamps (default = off)",
               "0");
   option_parser_register(opp, "-tm_temporal_bloomfilter_size", OPT_UINT32, &m_temporal_bloomfilter_size, 
               "size of bloomfilter for temporal conflict detection",
               "8192");
   option_parser_register(opp, "-tm_temporal_bloomfilter_n_hash", OPT_UINT32, &m_temporal_bloomfilter_n_hash, 
               "number of hashes in bloomfilter for temporal conflict detection",
               "4");
   option_parser_register(opp, "-tm_temporal_cd_addr_granularity", OPT_UINT32, &m_temporal_cd_addr_granularity, 
               "address granularity (in bytes) of temporal conflict detection",
               "4");

   // add itself to the checkers 
   option_parser_add_checker(opp, this); 
}

tm_bloomfilter_options::tm_bloomfilter_options()
   : m_init(false), m_size(0), m_n_hashes(0)
{ }

void tm_bloomfilter_options::reg_options(option_parser_t opp)
{
   option_parser_register(opp, "-tm_bloomfilter_sig_size", OPT_UINT32, &m_size, 
               "size of the bloomfilter for tm conflict detection (default = 64)",
               "64");
   option_parser_register(opp, "-tm_bloomfilter_n_hashes", OPT_UINT32, &m_n_hashes, 
               "number of hash functions used in the bloomfilter for tm conflict detection (default = 4)",
               "4");
   tm_bloomfilter_hashed_core_set::reg_options(opp); 
   bloomfilter_reg_options(opp); 
   init(); 
}

void tm_bloomfilter_options::init()
{
   if (m_init == false) {
      m_funct_ids.assign(4,0); 
      m_funct_ids[0] = 0; 
      m_funct_ids[1] = 1; 
      m_funct_ids[2] = 2; 
      m_funct_ids[3] = 3; 
      m_init = true; 
   }
}

tm_bloomfilter_set::tm_bloomfilter_set(bool counter_based)
   : m_access_set(g_tm_bloomfilter_options.m_size, g_tm_bloomfilter_options.m_funct_ids, g_tm_bloomfilter_options.m_n_hashes, counter_based), 
     m_write_set(g_tm_bloomfilter_options.m_size, g_tm_bloomfilter_options.m_funct_ids, g_tm_bloomfilter_options.m_n_hashes, counter_based),
     m_core_bf(NULL), m_core_id(-1)
{ }

tm_bloomfilter_core_set::tm_bloomfilter_core_set()
   : tm_bloomfilter_set(true)
{ }

void tm_bloomfilter_core_set::update_access_bloomfilter(const std::vector<int>& update_positions) 
{
   m_access_set.set_bitpos(update_positions); 
}

void tm_bloomfilter_core_set::remove_access_set(const bloomfilter &thread_access_set) 
{
   m_access_set.remove(thread_access_set); 
}

void tm_bloomfilter_set::mem_access(bool rd, unsigned addr, unsigned nbytes)
{
   unsigned word_size_log2 = g_tm_options.m_word_size_log2; 
   unsigned nwords = nbytes >> word_size_log2; 
   if ((nbytes % (1 << word_size_log2)) != 0) 
      nwords += 1; 
   unsigned word_addr = addr >> word_size_log2; 

   // update access set 
   for (int n = 0; n < nwords; n++) {
      std::vector<int> update_positions(m_access_set.n_hashes(), hashtable::s_nullpos); 
      m_access_set.add(word_addr + n, update_positions); 
      if (m_core_bf != NULL) {
         m_core_bf->update_access_bloomfilter(update_positions); 
      }
   }

   // update write set as well for write 
   if (not rd) {
      for (int n = 0; n < nwords; n++) 
         m_write_set.add(word_addr + n); 
   }
}

bool tm_bloomfilter_set::match_access_set(const tm_bloomfilter_set& other)
{
   return m_access_set.match(other.m_write_set); 
}

bool tm_bloomfilter_set::match_access_set(const addr_set_t& w_set) 
{
   addr_set_t::const_iterator i_waddr = w_set.begin(); 
   for (; i_waddr != w_set.end(); ++i_waddr) {
      if (m_access_set.match(*i_waddr)) {
         return true; 
      }
   } 
   return false; 
}

void tm_bloomfilter_set::clear()
{
   if (m_core_bf != NULL) {
      m_core_bf->remove_access_set(m_access_set); 
   }
   m_access_set.clear(); 
   m_write_set.clear(); 
}

unsigned int tm_bloomfilter_hashed_core_set::s_thread_hash_size = 32; 

tm_bloomfilter_hashed_core_set::tm_bloomfilter_hashed_core_set()
   : m_access_set_mt(g_tm_bloomfilter_options.m_size, g_tm_bloomfilter_options.m_funct_ids, g_tm_bloomfilter_options.m_n_hashes, 2048, s_thread_hash_size), m_core_id(-1)
{ }

void tm_bloomfilter_hashed_core_set::reg_options(option_parser_t opp)
{
   option_parser_register(opp, "-tm_bloomfilter_thread_hash_size", OPT_UINT32, &s_thread_hash_size, 
               "size of hashed thread vector in the memory-side bloomfilter for tm conflict detection (default = 32)",
               "32");
}

void tm_bloomfilter_hashed_core_set::mem_access(int hw_thread_id, bool rd, unsigned addr, unsigned nbytes)
{
   const unsigned word_size = g_tm_options.m_word_size; 
   const unsigned word_size_log2 = g_tm_options.m_word_size_log2; 

   unsigned nwords = nbytes >> word_size_log2; 
   if ((nbytes % word_size) != 0) 
      nwords += 1; 
   unsigned word_addr = addr >> word_size_log2; 

   // update access set 
   m_access_set_mt.select_thread(hw_thread_id); 
   for (int n = 0; n < nwords; n++) {
      m_access_set_mt.add(word_addr + n); 
   }
   m_access_set_mt.unselect_thread(); 
}

unsigned int tm_bloomfilter_hashed_core_set::match_access_set(const addr_set_t& w_set, hashtable_bits_mt::tvec_t &matched_threads)
{
   const bool non_hash_match = false; 
   addr_set_t::const_iterator i_waddr = w_set.begin(); 
   if (non_hash_match) {
      for (; i_waddr != w_set.end(); ++i_waddr) {
         if (m_access_set_mt.match(*i_waddr)) // use non-hashed thread vector 
            return 1; 
      } 
      return 0; 
   } else {
      matched_threads.reset(); 
      for (; i_waddr != w_set.end(); ++i_waddr) {
         m_access_set_mt.match_hashed(*i_waddr, matched_threads); // use hashed thread vector 
      } 
      return matched_threads.count(); 
   }
}

void tm_bloomfilter_hashed_core_set::clear(int hw_thread_id)
{
   m_access_set_mt.select_thread(hw_thread_id); 
   m_access_set_mt.clear(); 
   m_access_set_mt.unselect_thread(); 
}

void tm_bloomfilter_hashed_core_set::clear_all()
{
   m_access_set_mt.clear_all(); 
}

///////////////////////////////////////////////////////////////////////////////
// TM ring conflict detection implementation 

tm_ring_commit_record::tm_ring_commit_record(unsigned tuid, const addr_set_t& write_set, int commit_time)
   : m_tuid(tuid), m_timestamp(commit_time), m_write_set(write_set), m_status(tm_ring::COMPLETE), 
     m_write_filter(g_tm_bloomfilter_options.m_size, g_tm_bloomfilter_options.m_funct_ids, 
                    g_tm_bloomfilter_options.m_n_hashes, false)
{ 
   for (addr_set_t::const_iterator iAddr = m_write_set.begin(); iAddr != m_write_set.end(); ++iAddr) {
      m_write_filter.add(*iAddr); 
   }
}

// match against a given read set 
bool tm_ring_commit_record::match(const addr_version_set_t& other_tx_read_set) const 
{
   addr_version_set_t::const_iterator iAddr = other_tx_read_set.begin(); 
   for (; iAddr != other_tx_read_set.end(); ++iAddr) {
      if (iAddr->second >= m_timestamp) continue; // version of this word is newer than current timestamp -- skip check
      if (m_write_set.find(iAddr->first) != m_write_set.end()) {
         return true;
      }
   }
   return false; 
}

// match the write filter against a given read set 
bool tm_ring_commit_record::match_filter(const addr_version_set_t& other_tx_read_set) const 
{
   addr_version_set_t::const_iterator iAddr = other_tx_read_set.begin(); 
   for (; iAddr != other_tx_read_set.end(); ++iAddr) {
      if (iAddr->second >= m_timestamp) continue; // version of this word is newer than current timestamp -- skip check
      if ( m_write_filter.match(iAddr->first) ) {
         return true;
      }
   }
   return false; 
}

void tm_ring_commit_record::print(FILE *fout) const 
{
   fprintf(fout, "[%5d]", m_timestamp); 
   m_write_filter.print(fout); 
}


tm_ring_global::tm_ring_global()
   : m_ring_index(0)
{
   // jumpstart the ring with a empty commit record 
   addr_set_t empty_write_set; 
   tm_ring_commit_record new_record(-1, empty_write_set, 0); 
   m_ring.push_back(new_record); 
}

// commit transaction to ring 
void tm_ring_global::commit_tx(unsigned tuid, const addr_set_t& write_set) 
{
   int record_capacity = g_tm_options.m_ring_tm_record_capacity;
   if (record_capacity == 0 or write_set.size() <= record_capacity) {
      // write set can be hold with a single commit record 
      commit_tx_single_record(tuid, write_set); 
   } else {
      // split up write set among multiple commit records 
      addr_set_t partial_write_set; 
      // spread write set even among records 
      int chunks = write_set.size() / record_capacity + ((write_set.size() % record_capacity)? 1 : 0); 
      int record_size = write_set.size() / chunks + ((write_set.size() % chunks)? 1 : 0); 
      for (addr_set_t::const_iterator iAddr = write_set.begin(); iAddr != write_set.end(); ++iAddr) {
         partial_write_set.insert(*iAddr); 
         if (partial_write_set.size() >= record_size) {
            commit_tx_single_record(tuid, partial_write_set); 
            partial_write_set.clear(); 
         }
      }
      if (partial_write_set.empty() == false) {
         commit_tx_single_record(tuid, partial_write_set); 
      }
   }
}

void tm_ring_global::commit_tx_single_record(unsigned tuid, const addr_set_t& write_set) 
{
   // push record into compressed ring as well 
   if (g_tm_options.m_compressed_ring_capacity) {
      unsigned int comp_ring_size = g_tm_options.m_compressed_ring_capacity;
      if (m_ring_index % comp_ring_size == 0) {
         tm_ring_compressed_record comprec(m_ring_index, comp_ring_size, g_tm_options.m_compressed_ring_size); 
         m_compressed_ring.push_back(comprec); 
      }
      int comp_ring_index = m_ring_index / comp_ring_size;
      m_compressed_ring[comp_ring_index].commit_record(write_set, m_ring_index); 
   }

   int old_ring_index = m_ring_index; 
   m_ring_index += 1; 
   assert(m_ring_index > old_ring_index); // overflow detection 

   // assume write is instantaneous, so there is no WRITING state 
   // so just create a new record and push it into the ring 
   tm_ring_commit_record new_record(tuid, write_set, m_ring_index); 
   m_ring.push_back(new_record); 

   assert(m_ring_index + 1 == (int)m_ring.size()); 
}

// check if any of the commit records from ring index downto start_time + 1 conflicts with given read set 
// return start_time if there is no conflict 
int tm_ring_global::check_conflict(const addr_version_set_t& other_tx_read_set, int start_time) const 
{
   assert(start_time >= 0); 
   int rec; 
   bool conflict_detected = false; 
   int ring_size_limit = 0;
   if (g_tm_options.m_ring_tm_size_limit != 0) {
      ring_size_limit = m_ring_index - g_tm_options.m_ring_tm_size_limit; // only a subset of entries are alive
      if (ring_size_limit < 0) ring_size_limit = 0; 
   }

   for (rec = m_ring_index; rec > start_time; rec--) {
      bool ideal_match = m_ring[rec].match(other_tx_read_set); 
      if (ideal_match == true or rec < ring_size_limit) {
         conflict_detected = true; 
         break; 
      }
   }

   // conflict detection with bloom filter in each record 
   if (g_tm_options.m_ring_tm_bloomfilter) {
      int bf_rec; 
      bool bf_conflict_detected = false; 
      for (bf_rec = m_ring_index; bf_rec > start_time; bf_rec--) {
         bool bloomfilter_match = m_ring[bf_rec].match_filter(other_tx_read_set);
         if (bloomfilter_match == true or rec < ring_size_limit) {
            bf_conflict_detected = true; 
            break; 
         }
      }
      rec = bf_rec; 
      if (conflict_detected == false and bf_conflict_detected == true) {
         g_tm_ring_stats.m_ring_bloomfilter_false_conflicts += 1; 
      } else {
         assert(not(conflict_detected and not bf_conflict_detected)); 
      }
   }

   // conflict detection with compressed record 
   if (g_tm_options.m_compressed_ring_capacity) {
      bool cring_conflict_detected = false; 
      for (compressed_ring_t::const_reverse_iterator iCRing = m_compressed_ring.rbegin(); 
           iCRing != m_compressed_ring.rend(); ++iCRing) 
      {
         bool cring_match = iCRing->match(other_tx_read_set, start_time); 
         if (cring_match) {
            cring_conflict_detected = true; 
            break; 
         }
      }
      if (cring_conflict_detected) {
         rec = start_time + 1; //indicating a conflict 
      } else {
         assert(not(conflict_detected and not cring_conflict_detected)); 
      }
   }

   g_tm_ring_stats.m_commit_rec_distance.add2bin(m_ring_index - start_time); 
   if (conflict_detected) {
      g_tm_ring_stats.m_commit_rec_dist_conflict.add2bin(m_ring_index - start_time); 
      g_tm_ring_stats.m_commit_rec_actual_conflict_distance.add2bin(m_ring_index - rec); 
   } else {
      g_tm_ring_stats.m_commit_rec_dist_no_conflict.add2bin(m_ring_index - start_time); 
   }

   return rec; 
}

void tm_ring_global::print(FILE *fout) const 
{
   for (ring_t::const_iterator irec = m_ring.begin(); irec != m_ring.end(); ++irec) {
      irec->print(fout); 
   }
}

tm_ring_compressed_record::tm_ring_compressed_record(int base_index, unsigned int n_records, unsigned int compressed_size)
   : m_base_index(base_index), m_next_rec_index(0), m_n_records(n_records), m_compressed_size(compressed_size),
     m_write_filter(g_tm_bloomfilter_options.m_size, g_tm_bloomfilter_options.m_funct_ids, 
                    g_tm_bloomfilter_options.m_n_hashes, m_n_records, compressed_size)
{ }

void tm_ring_compressed_record::commit_record(const addr_set_t& write_set, int commit_time)
{
   int internal_index = commit_time - m_base_index; 
   assert(internal_index < m_n_records); 
   assert(internal_index == m_next_rec_index); 
   m_write_filter.select_thread(internal_index); 
   addr_set_t::const_iterator iAddr;
   for (iAddr = write_set.begin(); iAddr != write_set.end(); ++iAddr) {
      m_write_filter.add(*iAddr); 
   }
   m_write_filter.unselect_thread(); 
   m_next_rec_index += 1; 
}

bool tm_ring_compressed_record::match(const addr_version_set_t& other_tx_read_set, int starttime) const
{
   addr_version_set_t::const_iterator iAddr = other_tx_read_set.begin(); 
   for (; iAddr != other_tx_read_set.end(); ++iAddr) {
      int addr_version = std::max(starttime, iAddr->second); 
      if (addr_version >= (m_base_index + m_n_records)) continue; // version of this word is newer than latest timestamp 
      hashtable_bits_mt::tvec_t matched_records; 
      if ( m_write_filter.match_hashed(iAddr->first, matched_records) ) {
         if (addr_version < m_base_index) {
            return true;
         } else {
            int internal_start = addr_version - m_base_index; 
            for (int idx = internal_start; idx < m_next_rec_index; idx++) {
               if (matched_records.test(idx) == true) 
                  return true; 
            }
         }
      }
   }
   return false; 
}

void tm_ring_stats::print(FILE *fout)
{
    m_commit_rec_distance.fprint(fout); fprintf(fout, "\n"); 
    m_commit_rec_dist_conflict.fprint(fout); fprintf(fout, "\n"); 
    m_commit_rec_dist_no_conflict.fprint(fout); fprintf(fout, "\n"); 
    m_commit_rec_actual_conflict_distance.fprint(fout); fprintf(fout, "\n"); 
    fprintf(fout, "tm_ring_bloomfilter_false_conflicts = %u\n", m_ring_bloomfilter_false_conflicts); 
}

///////////////////////////////////////////////////////////////////////////////
// Temporal conflict detection 

temporal_conflict_detector::temporal_conflict_detector() 
{
   std::vector<int> func_ids(4); 
   func_ids[0] = 0; 
   func_ids[1] = 1; 
   func_ids[2] = 2; 
   func_ids[3] = 3; 
   m_rbloomfilter = new versioning_bloomfilter(g_tm_options.m_temporal_bloomfilter_size, func_ids, g_tm_options.m_temporal_bloomfilter_n_hash); 
}

temporal_conflict_detector::~temporal_conflict_detector() 
{
   delete m_rbloomfilter; 
}

// function to standardize granularity of addresses 
addr_t temporal_conflict_detector::get_chunk_address( addr_t input_addr ) 
{
   // assume input address are word-sized (already shifted by 2 bits)
   addr_t bits_to_ignore = g_tm_options.m_temporal_cd_addr_granularity_log2 - 2; 

   return (input_addr >> bits_to_ignore); 
}

// return with a timestamp indicating when the word can be last written 
tm_timestamp_t temporal_conflict_detector::at_transaction_read( addr_t addr ) 
{
   addr_t chunk_addr = get_chunk_address(addr); 
   if (g_tm_options.m_exact_temporal_conflict_detection) {
      tm_timestamp_t exact_time = m_last_written_timetable[chunk_addr]; 
      tm_timestamp_t approx_time = m_rbloomfilter->get_version(chunk_addr); 
      assert(approx_time >= exact_time); // conservativeness check 
      return exact_time; 
   } else {
      tm_timestamp_t approx_time = m_rbloomfilter->get_version(chunk_addr); 
      return approx_time; 
   }
}

// update the last written time of a word when it is updated at commit 
void temporal_conflict_detector::update_word( addr_t addr, tm_timestamp_t new_time) 
{
   addr_t chunk_addr = get_chunk_address(addr); 
   if (g_tm_options.m_exact_temporal_conflict_detection) {
      tm_timestamp_t old_time = m_last_written_timetable[chunk_addr]; 
      assert(old_time <= new_time); // detect potential overflow 
      m_last_written_timetable[chunk_addr] = new_time; 
   }

   m_rbloomfilter->update_version(chunk_addr, new_time); 
}

void temporal_conflict_detector::dump( FILE *fp )
{
   fprintf(fp, "Exact last written time:\n"); 
   for (auto iword = m_last_written_timetable.begin(); iword != m_last_written_timetable.end(); ++iword) {
      fprintf(fp, "[0x%08x] at cycle %llu \n", iword->first, iword->second); 
   }
}

// global singleton
temporal_conflict_detector * temporal_conflict_detector::s_temporal_conflict_detector = NULL;

temporal_conflict_detector& temporal_conflict_detector::get_singleton() 
{
   if (s_temporal_conflict_detector == NULL) 
      s_temporal_conflict_detector = new temporal_conflict_detector(); 

   return *s_temporal_conflict_detector; 
}


///////////////////////////////////////////////////////////////////////////////
// Value-based conflict detection 

value_based_tm_manager::value_based_tm_manager( ptx_thread_info *thread, bool timing_mode )
   : tm_manager(thread, timing_mode), m_violated(false), m_gmem(NULL), m_last_validation(0),
     m_n_reread(0), m_n_reread_violation(0), m_n_timeout(0), m_warp_level_raw(0) 
{ }

value_based_tm_manager::~value_based_tm_manager()
{ }

void value_based_tm_manager::at_start() 
{ 
   m_last_validation = gpu_sim_cycle; 
}

bool value_based_tm_manager::at_access( memory_space *mem, bool potential_conflicting, bool rd, addr_t addr, int nbytes )
{
   // check for timeout, if it is, validate and self-abort if required 
   if (g_tm_options.m_timeout_validation > 0 and (gpu_sim_cycle - m_last_validation) > g_tm_options.m_timeout_validation) {
      m_n_timeout++; 
      validate(); 
      if (m_violated) 
         return true; 
      else 
         m_last_validation = gpu_sim_cycle; // reset the timeout 
   }

   // only concern about potentially conflicting reads 
   if (potential_conflicting == false or rd == false) return false; 

   if (m_gmem == NULL) m_gmem = mem; 

   addr_t word_size = g_tm_options.m_word_size; 
   addr_t word_size_log2 = g_tm_options.m_word_size_log2; 

   addr_t base_waddr = addr >> word_size_log2;
   addr_t limit_waddr = (addr + nbytes) >> word_size_log2;
   if (((addr + nbytes) & (word_size - 1)) != 0) 
      limit_waddr += 1; // account for non-word-aligned access 

   for (addr_t waddr = base_waddr; waddr < limit_waddr; waddr++) {
      // skip if this memory location is already part of write-set 
      if (m_write_word_set.find(waddr) != m_write_word_set.end()) continue; 
      // same check for warp-level transactions 
      if (m_gmem_view_tx != NULL) {
         auto other_tx = dynamic_cast<value_based_tm_manager*>(m_gmem_view_tx); 
         auto warp_level_write_word_set = &(other_tx->m_warp_level_write_word_set); 
         if (warp_level_write_word_set->find(waddr) != warp_level_write_word_set->end()) {
            m_warp_level_raw++; 
            continue; 
         }
      }

      // read value from memory 
      unsigned int mem_value = 0; 
      mem->read(waddr << word_size_log2, word_size, &mem_value); 
      addr_value_t::iterator iAdValue = m_read_set_value.find(waddr); 
      if (iAdValue != m_read_set_value.end()) {
         m_n_reread += 1; 
         if (iAdValue->second != mem_value) {
            m_violated = true; 
            m_n_reread_violation += 1; 
         }
      } else {
         // this is a new read, buffer the mem value  
         m_read_set_value[waddr] = mem_value; 
      }

      // temporal conflict detection 
      // obtain the last written time + set the first read time 
      tm_timestamp_t last_written_time = temporal_conflict_detector::get_singleton().at_transaction_read(waddr); 
      m_temporal_cd_metadata.update_last_written_time(last_written_time); 
      if (m_temporal_cd_metadata.m_first_read_done == false) {
         m_temporal_cd_metadata.set_first_read_time(gpu_sim_cycle + gpu_tot_sim_cycle); 
      }
   }

   if (g_tm_options.m_value_based_eager_cr) {
      // eager conflict resolution - self-abort as soon as violated  
      return m_violated; 
   } else {
      // lazy conflict resolution 
      return false; 
   }
}

void value_based_tm_manager::at_abort()
{
   m_read_set_value.clear(); 
   m_violated = false; 
   m_gmem = NULL; 
   m_n_reread = 0;
   m_warp_level_raw = 0; 
   m_temporal_cd_metadata.reset(); 
}

// check for read-set consistency one last time, self-abort if it is not consistent
// return true if self-aborted 
bool value_based_tm_manager::at_commit_validation()
{
   validate(); 

   return m_violated; 
}

void value_based_tm_manager::at_commit_success()
{ 
   g_tm_global_statistics.m_n_reread.add2bin(m_n_reread); 
   g_tm_global_statistics.m_n_reread_violation += m_n_reread_violation; 
   g_tm_global_statistics.m_n_timeout_validation.add2bin(m_n_timeout); 
   g_tm_global_statistics.m_n_warp_level_raw.add2bin(m_warp_level_raw); 
}

void value_based_tm_manager::validate() 
{
   addr_t word_size = g_tm_options.m_word_size; 
   addr_t word_size_log2 = g_tm_options.m_word_size_log2; 

   // run though the whole read set and ensure that the buffered value is not changed in memory 
   for (addr_value_t::const_iterator iAdValue = m_read_set_value.begin(); 
        iAdValue != m_read_set_value.end(); ++iAdValue) 
   {
      unsigned int mem_value = 0; 
      m_gmem->read(iAdValue->first << word_size_log2, word_size, &mem_value); 
      if (iAdValue->second != mem_value) {
         m_violated = true; 
      }
      if (watched()) {
         printf("[TMM-%llu] Thd %u timeout-validates addr[%#08x]=%#x see %#x in GMem\n",
                gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid, iAdValue->first << word_size_log2, iAdValue->second, mem_value);
      }
   }
}

bool value_based_tm_manager::validate_all( bool useTemporalCD ) 
{
   bool conflict_exist_via_tcd = m_temporal_cd_metadata.conflict_exist(); 
   validate(); 
   if (m_violated and not conflict_exist_via_tcd) {
      // Uncaught conflict via temporal conflict detection
      g_tm_global_statistics.m_n_vcd_tcd_mismatch += 1; 
   } 
   if (useTemporalCD) {
      return (not conflict_exist_via_tcd); 
   } else {
      return (not m_violated); 
   }
}

void value_based_tm_manager::dump_read_set()
{
   addr_t word_size = g_tm_options.m_word_size; 
   addr_t word_size_log2 = g_tm_options.m_word_size_log2; 

   printf("Read Set Value:\n"); 
   // run though the whole read set and ensure that the buffered value is not changed in memory 
   for (addr_value_t::const_iterator iAdValue = m_read_set_value.begin(); 
        iAdValue != m_read_set_value.end(); ++iAdValue) 
   {
      addr_t vAddr = iAdValue->first << word_size_log2; 
      unsigned int vVal = iAdValue->second; 

      unsigned int mem_value = 0; 
      m_gmem->read(vAddr, word_size, &mem_value); 
      printf("addr[%#08x] = %x (%x @ Mem)\n", vAddr, vVal, mem_value); 
   }
}

void value_based_tm_manager::dump_committed_set()
{
   addr_t word_size = g_tm_options.m_word_size; 

   printf("Committed Set:\n"); 
   // run though the whole read set and ensure that the buffered value is not changed in memory 
   for (addr_set_t::const_iterator iAddr = m_committed_set.begin(); 
        iAddr != m_committed_set.end(); ++iAddr) 
   {
      addr_t vAddr = *iAddr; 

      unsigned int mem_value = 0; 
      m_gmem->read(vAddr, word_size, &mem_value); 
      printf("addr[%#08x] = %x @ Mem\n", vAddr, mem_value); 
   }
}

// for debugging data race -- uncomment next line to activate
// #define track_last_writer 
struct last_writer_info {
   unsigned m_thread_uid; 
   unsigned m_thread_sc;
   unsigned m_thread_hwtid; 
   unsigned long long m_time; 
   last_writer_info() 
      : m_thread_uid(0), m_thread_sc(0), m_thread_hwtid(0), m_time(0) 
   { }
   last_writer_info(unsigned thread_uid, unsigned sc, unsigned hwtid, unsigned long long time) 
      : m_thread_uid(thread_uid), m_thread_sc(sc), m_thread_hwtid(hwtid), m_time(time) 
   { }
};
tr1_hash_map<addr_t, last_writer_info> g_last_writer; 

// interface for timing model validation and commit -- only for value-based tm

// for a given word, return true if the buffered value in TM manager is still consistent with global memory 
bool value_based_tm_manager::validate_addr( addr_t addr ) 
{ 
   addr_t word_size = g_tm_options.m_word_size; 
   addr_t word_size_log2 = g_tm_options.m_word_size_log2; 

   addr_value_t::const_iterator iAdValue; 
   iAdValue = m_read_set_value.find(addr >> word_size_log2); 

   dim3 ctaid = m_thread->get_ctaid(), tid = m_thread->get_tid();

   if (getenv("TOMMYDBG") || iAdValue == m_read_set_value.end()) {
	   printf("[value_based_tm_manager::validate_addr] Thread (%u,%u,%u)-(%u,%u,%u) Read Set (words):\n",
		   ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z);
	   for (addr_value_t::iterator itr = m_read_set_value.begin();
		   itr != m_read_set_value.end(); itr++) {
		   printf("Word %08x (Addr %08x) = %x\n", itr->first, (itr->first)<<word_size_log2, itr->second);
	   }
   }

   if(iAdValue == m_read_set_value.end()) {
	   printf("But addr is %08x\n", addr);
	   printf("Thread (%u,%u,%u)-(%u,%u,%u) UID=%u\n",
			ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z, m_thread->get_uid());

	   shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_thread->m_core);
	   if (scc) {
		   struct tm_warp_info* twi = &(scc->m_warp[m_thread->m_hw_wid].get_tm_warp_info());
		   int lane_id = m_thread->get_hw_tid() % 32;
		   printf("RW Log in TM Warp Info (Lane %d):\n", lane_id);
		   for (int i=0; i<twi->m_read_log_size; i++) {
			   addr_t x = twi->m_read_log[i].m_addr[lane_id];
			   if (x != 0) {
				   printf("ReadLog[%d] = %x\n", i, x);
			   }
		   }
	   }
	   assert(0);
   }

   unsigned int mem_value = 0;
   m_gmem->read(addr, word_size, &mem_value); 

   if (watched()) {
      printf("[TMM-%llu] Thd %u validates addr[%#08x]=%#x see %#x in GMem\n", 
             gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid, addr, iAdValue->second, mem_value);
   }

   return (iAdValue->second == mem_value); 
}

void value_based_tm_manager::commit_addr( addr_t addr ) 
{
   // skip if doing all commit on core side 
   if (g_tm_options.m_timing_mode_core_side_commit == true) return;

   // validate_or_crash(); 

   // if this is part of read-set, before overwritting it, validate the value again
   addr_t word_size = g_tm_options.m_word_size; 
   addr_t word_size_log2 = g_tm_options.m_word_size_log2; 
   addr_value_t::const_iterator iAdValue; 
   iAdValue = m_read_set_value.find(addr >> word_size_log2); 
   if (m_committed_set.find(addr) == m_committed_set.end() and iAdValue != m_read_set_value.end()) {
      unsigned int mem_value = 0;
      m_gmem->read(addr, word_size, &mem_value); 
      assert (iAdValue->second == mem_value); 
   }

   const unsigned cu_word_size = 4; 

   // search in reverse order to get the most updated value for the same word 
   bool commit_done = false; 
	std::list<access_record>::reverse_iterator i;
	for( i=m_write_data.rbegin(); i != m_write_data.rend(); i++ ) { 
		access_record &w = *i;
      if (w.contain_addr(addr, cu_word_size) and w.get_memory_space() == m_gmem) {
         // w.commit();
         w.commit_word(addr, cu_word_size);
         if (watched()) {
            printf("[TMM-%llu] Thd %u commits addr[%#08x]=", 
                   gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid, addr);
            w.print(stdout);
            printf("\n"); 
         }
         #ifdef track_last_writer
         g_last_writer[addr] = last_writer_info(m_thread_uid, m_thread_sc, m_thread_hwtid, gpu_sim_cycle + gpu_tot_sim_cycle); 
         #endif
         commit_done = true; 
         break; 
      }
	}

   addr_t waddr = addr >> word_size_log2; 
   temporal_conflict_detector::get_singleton().update_word( waddr, gpu_sim_cycle + gpu_tot_sim_cycle ); 
   assert(commit_done == true); 
   m_committed_set.insert(addr); 
}

// ensure the committing transaction is still consistent 
void value_based_tm_manager::validate_or_crash( ) 
{  
   addr_t word_size = g_tm_options.m_word_size; 
   addr_t word_size_log2 = g_tm_options.m_word_size_log2; 

   // run though the whole read set and ensure that the buffered value is not changed in memory 
   for (addr_value_t::const_iterator iAdValue = m_read_set_value.begin(); 
        iAdValue != m_read_set_value.end(); ++iAdValue) 
   {
      // do not compare values that are committed by this very transaction 
      addr_t vAddr = iAdValue->first << word_size_log2; 
      unsigned int vVal = iAdValue->second; 
      if (m_committed_set.find(vAddr) == m_committed_set.end()) {
         unsigned int mem_value = 0; 
         m_gmem->read(vAddr, word_size, &mem_value); 
         if (vVal != mem_value) {
            #ifdef track_last_writer
            last_writer_info &last_writer = g_last_writer.find(vAddr)->second; 
            printf("[TMM-%llu] Validation protection failed @ %#08x: expects %x but sees %x, last writer=(%u,%u,%u)-%llu\n", 
                   gpu_sim_cycle + gpu_tot_sim_cycle, vAddr, vVal, mem_value, 
                   last_writer.m_thread_uid, last_writer.m_thread_sc, last_writer.m_thread_hwtid, last_writer.m_time ); 
            #endif
            dim3 ctaid = m_thread->get_ctaid(), tid = m_thread->get_tid();
			printf("[TMM-%llu] Validation protection failed (%u,%u,%u)-(%u,%u,%u) (UID=%u) @ %#08x: expects %x but sees %x\n",
				   gpu_sim_cycle + gpu_tot_sim_cycle,
				   ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z, m_thread->get_uid(),
				   vAddr, vVal, mem_value);
			printf("Read set (words) (value_based_tm_manager::m_read_set_value): \n");
			for (addr_value_t::const_iterator itr = m_read_set_value.begin();
			        itr != m_read_set_value.end(); ++itr)
				printf(" [%#08x] = %x\n", 4*itr->first, itr->second);
			printf("# of aborted/samewarp/all staggers = %u / %u\n", g_num_stagger_aborts, g_num_stagger_samewarp, g_num_staggers);
			shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_thread->get_core());
			if (scc) {
				const unsigned wid = m_thread->get_tid().x / 32;
				scc->m_scoreboard->printContents();
				scc->m_simt_stack[wid]->print(stdout);
				tm_warp_info& twi = scc->m_warp[wid].get_tm_warp_info();
				printf("This warp's backup R/W Set Stack Size: %lu\n", twi.backup_stack.size());
				tx_log_walker_warpc* tlwc = dynamic_cast<tx_log_walker_warpc*>(scc->m_ldst_unit->m_TLW);
				if (tlwc) {
					printf("This committing warp's thread state: \n");
					tx_log_walker_warpc::warp_commit_tx_t& cmt_warp = tlwc->m_committing_warp[wid];
					for (unsigned t = 0; t < 32; t++) {
						cmt_warp.m_thread_state[t].print(stdout);
					}
				}
			}
            assert(0); 
         }
      }
   }
}

void value_based_tm_manager::commit_core_side( ) 
{
   // for error checking 
   // validate(); 
   // assert(m_violated == false); 
   if (m_write_data.empty() == false) // only check for writing transaction
      validate_or_crash(); 

   if (watched()) {
      printf("[TMM-%llu] Thd %u Commit @ Core-Side\n", gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid );
   }

   bool commit_all = g_tm_options.m_timing_mode_core_side_commit; 
	// step 1: commit updates to memory
	std::list<access_record>::iterator i;
	for( i=m_write_data.begin(); i != m_write_data.end(); i++ ) {
      // limit this to just the accesses to non-global memory 
		access_record &w = *i;
      if (w.get_memory_space() != m_gmem or commit_all) {
         if (watched()) {
            printf("[TMM-%llu] Thd %u commits addr[%#08x]=", 
                   gpu_sim_cycle + gpu_tot_sim_cycle, m_thread_uid, w.getaddr());
            w.print(stdout);
            printf("\n"); 
         }
         w.commit();
         #ifdef track_last_writer
         g_last_writer[w.getaddr()] = last_writer_info(m_thread_uid, m_thread_sc, m_thread_hwtid, gpu_sim_cycle + gpu_tot_sim_cycle); 
         #endif
      }
	}
   bool writing_tx = not m_write_data.empty(); 

   m_thread->tm_commit();
   g_tm_global_statistics.m_regs_buffered_max.add2bin(m_thread->m_tm_regs_buffered_max);
   g_tm_global_statistics.m_regs_modified_max.add2bin(m_thread->m_tm_regs_modified_max);
   g_tm_global_statistics.m_regs_read_max.add2bin(m_thread->m_tm_regs_read_max);

   // step 2: update global state -- commit is successful 
   at_commit_success(); 

	// m_thread->end_transaction();

   // update statistics 
	g_tm_global_statistics.m_n_commits += 1;
   if (writing_tx) g_tm_global_statistics.m_n_writing_commits += 1;
	g_tm_global_statistics.dec_concurrency();
   g_tm_global_statistics.record_commit_tx_size(m_read_word_set.size(), m_write_word_set.size(), m_access_word_set.size());
   g_tm_global_statistics.record_tx_blockcount(m_read_block_set, m_write_block_set, m_access_block_set); 
   g_tm_global_statistics.record_raw_info(m_raw_set.size(), m_raw_access); 
   g_tm_global_statistics.m_aborts_per_transaction.add2bin(m_abort_count); 
   g_tm_global_statistics.m_duration.add2bin(gpu_sim_cycle - m_start_cycle); 
   g_tm_global_statistics.m_duration_first_rd.add2bin(gpu_sim_cycle - m_first_read_cycle); 
   g_tm_global_statistics.m_write_buffer_footprint.add2bin(m_buffered_write_word_set.size()); 
   g_tm_global_statistics.m_n_read.add2bin(m_n_read); 
   g_tm_global_statistics.m_n_write.add2bin(m_n_write); 
   g_tm_global_statistics.m_n_rewrite.add2bin(m_n_rewrite / g_tm_options.m_word_size); 
   write_access_log(); 

#ifdef DEBUG_TM
	printf("[%Lu] finished committing transaction %u (uid=%u) for tid=(%u,%u,%u) cta=(%u,%u,%u) tuid=%u, sc=%u, hwtid=%u\n", 
		   gpu_sim_cycle,
		   m_thread->tm_num_transactions(), m_uid,
		   tid.x, tid.y, tid.z, ctaid.x, ctaid.y, ctaid.z, 
		   m_thread_uid, m_thread_sc, m_thread_hwtid );
#else
   if (g_tm_global_statistics.m_n_commits % TM_MSG_INV == 0) {
      g_tm_global_statistics.print_short(stdout); 
   }
#endif
}

// quickly compare two sorted set for non-null interaction 
bool fast_set_match(addr_set_t &a, addr_set_t &b) 
{
   addr_set_t::const_iterator first1 = a.cbegin(); 
   addr_set_t::const_iterator last1 = a.cend(); 
   addr_set_t::const_iterator first2 = b.cbegin(); 
   addr_set_t::const_iterator last2 = b.cend(); 

   bool match = false; 

   // code shamelessly ripped from STL::set_intersection
   while (first1!=last1 && first2!=last2)
   {
      if (*first1<*first2) ++first1;
      else if (*first2<*first1) ++first2;
      else {
         match = true; 
         ++first1; ++first2;
         return match; 
      }
   }
   return match;
}

bool value_based_tm_manager::has_conflict_with( tm_manager_inf * other_tx ) 
{
   value_based_tm_manager * otx = dynamic_cast<value_based_tm_manager*>(other_tx); 
   bool has_conflict = false; 

   bool has_rw_conflict = fast_set_match(m_read_word_set, otx->m_write_word_set); 
   bool has_wr_conflict = fast_set_match(m_write_word_set, otx->m_read_word_set); 
   has_conflict = has_rw_conflict or has_wr_conflict; 

   if (has_conflict) g_tm_global_statistics.m_n_intra_warp_detected_conflicts += 1; 

   return has_conflict; 
}



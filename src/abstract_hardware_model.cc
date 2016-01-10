// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh, Timothy Rogers,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



#include "abstract_hardware_model.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_sim.h"
#include "cuda-sim/ptx-stats.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/tm_manager.h"
#include "gpgpu-sim/gpu-sim.h"
#include "option_parser.h"
#include <algorithm>

unsigned mem_access_t::sm_next_access_uid = 0;   
unsigned warp_inst_t::sm_next_uid = 0;
extern tm_global_statistics g_tm_global_statistics;
extern unsigned g_stagger1_count, g_num_stagger_aborts;
extern bool g_tommy_dbg_0830, g_tommy_flag_0808_1;
extern unsigned g_0830_sanity_check_count, g_0830_n_count, g_0830_t_count;

void move_warp( warp_inst_t *&dst, warp_inst_t *&src )
{
   assert( dst->empty() );
   warp_inst_t* temp = dst;
   dst = src;
   src = temp;
   src->clear();
}

extern void CartographerTimeSeries_onTxBeginIssuedToFU(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst, const warp_inst_t* to_inst);
extern void CartographerTimeSeries_onWarpActiveMaskChanged(core_t* core, unsigned warp_id, const simt_mask_t* mask, const char* _reason);
extern address_type get_my_converge_point(address_type a);
extern int g_print_simt_stack_wid, g_print_simt_stack_sid, g_tommy_break_cycle;
extern unsigned g_watched_tid, g_tommy_reconverge_point_mode, g_0830_restore_by_pc_count;
extern bool g_tommy_dbg_0830;

class ptx_instruction;

static void PrintSIMTStackEntry (const std::bitset<MAX_WARP_SIZE_SIMT_STACK>* ety) {
	printf("[");
	if (!ety) { printf(" No stack entry "); }
	else {
		for (int i=0; i<ety->size(); i++) {
			if (ety->test(i)) { printf("1"); } else { printf("0"); }
		}
	}
	printf("]");
}

void gpgpu_functional_sim_config::reg_options(class OptionParser * opp)
{
	option_parser_register(opp, "-gpgpu_ptx_use_cuobjdump", OPT_BOOL,
                 &m_ptx_use_cuobjdump,
                 "Use cuobjdump to extract ptx and sass from binaries",
#if (CUDART_VERSION >= 4000)
                 "1"
#else
                 "0"
#endif
                 );
	option_parser_register(opp, "-gpgpu_experimental_lib_support", OPT_BOOL,
	                 &m_experimental_lib_support,
	                 "Try to extract code from cuda libraries [Broken because of unknown cudaGetExportTable]",
	                 "0");
    option_parser_register(opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL,
                 &m_ptx_convert_to_ptxplus,
                 "Convert SASS (native ISA) to ptxplus and run ptxplus",
                 "0");
    option_parser_register(opp, "-gpgpu_ptx_force_max_capability", OPT_UINT32,
                 &m_ptx_force_max_capability,
                 "Force maximum compute capability",
                 "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, 
                &g_ptx_inst_debug_to_file, 
                "Dump executed instructions' debug information to file", 
                "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file, 
                  "Executed instructions' debug output file",
                  "inst_debug.txt");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32, &g_ptx_inst_debug_thread_uid, 
               "Thread UID for executed instructions' debug output", 
               "1");
}

void gpgpu_functional_sim_config::ptx_set_tex_cache_linesize(unsigned linesize)
{
   m_texcache_linesize = linesize;
}

gpgpu_t::gpgpu_t( const gpgpu_functional_sim_config &config )
    : m_function_model_config(config)
{
   m_global_mem = new memory_space_impl<8192>("global",64*1024);
   m_tex_mem = new memory_space_impl<8192>("tex",64*1024);
   m_surf_mem = new memory_space_impl<8192>("surf",64*1024);

   m_dev_malloc=GLOBAL_HEAP_START; 

   if(m_function_model_config.get_ptx_inst_debug_to_file() != 0) 
      ptx_inst_debug_file = fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");
}

address_type line_size_based_tag_func(new_addr_type address, new_addr_type line_size)
{
   //gives the tag for an address based on a given line size
   return address & ~(line_size-1);
}

const char * mem_access_type_str(enum mem_access_type access_type)
{
   #define MA_TUP_BEGIN(X) static const char* access_type_str[] = {
   #define MA_TUP(X) #X
   #define MA_TUP_END(X) };
   MEM_ACCESS_TYPE_TUP_DEF
   #undef MA_TUP_BEGIN
   #undef MA_TUP
   #undef MA_TUP_END

   assert(access_type < NUM_MEM_ACCESS_TYPE); 

   return access_type_str[access_type]; 
}

void warp_inst_t::check_active_mask() const {
   // if a warp instruction active mask cleared due to effects other than predication
   if (m_warp_active_mask.any() == false and m_warp_predicate_mask.any() == false) {
      if (g_debug_execution >= 4)
         printf("[GPGPU-Sim] Active mask of the instruction from warp %d cleared\n", m_warp_id); 
      // assert(0); 
   }
}


void warp_inst_t::clear_active( const active_mask_t &inactive ) {
    active_mask_t test = m_warp_active_mask;
    test &= inactive;
    assert( test == inactive ); // verify threads being disabled were active
    m_warp_active_mask &= ~inactive;
    check_active_mask(); 
}

void warp_inst_t::predicate_off( unsigned lane_id ) {
    assert(m_warp_predicate_mask.test(lane_id) == false); 
    m_warp_predicate_mask.set(lane_id); 
    set_not_active(lane_id); 
}

void warp_inst_t::set_not_active( unsigned lane_id ) {
    m_warp_active_mask.reset(lane_id);
    std::list<mem_access_t>::iterator iAcc; 
    for (iAcc = m_accessq.begin(); iAcc != m_accessq.end(); ++iAcc) {
        iAcc->reset_lane(lane_id);
    }
    check_active_mask(); 
}

void warp_inst_t::set_stagger(unsigned lane_id) {
	assert (m_warp_active_mask.test(lane_id));
	m_warp_stagger_mask.set(lane_id);
	m_warp_active_mask.reset(lane_id);
}

void warp_inst_t::set_active( const active_mask_t &active ) {
   m_warp_active_mask = active;
   if( m_isatomic ) {
      for( unsigned i=0; i < m_config->warp_size; i++ ) {
         if( !m_warp_active_mask.test(i) ) {
             m_per_scalar_thread[i].callback.function = NULL;
             m_per_scalar_thread[i].callback.instruction = NULL;
             m_per_scalar_thread[i].callback.thread = NULL;
         }
      }
   }
   check_active_mask(); 
}

void warp_inst_t::do_atomic(bool forceDo) {
    do_atomic( m_warp_active_mask,forceDo );
}

void warp_inst_t::do_atomic( const active_mask_t& access_mask,bool forceDo ) {
    assert( m_isatomic && (!m_empty||forceDo) );
    for( unsigned i=0; i < m_config->warp_size; i++ )
    {
        if( access_mask.test(i) ) {
            dram_callback_t &cb = m_per_scalar_thread[i].callback;
            if( cb.thread )
                cb.function(cb.instruction, cb.thread);
        }
    }
}

// print out all accesses generated by this warp instruction 
void warp_inst_t::dump_access( FILE *fout ) const
{
   for (std::list<mem_access_t>::const_iterator i_access = m_accessq.begin(); i_access != m_accessq.end(); i_access++) {
      i_access->print(fout); 
      fprintf(fout, "\n");
   }
}

void warp_inst_t::generate_mem_accesses(tm_warp_info& warp_info)
{
	// 2015-08-12: How can this thing be 5 at PC=180 , ld.global.s32 ???
	// b/c "last memory space" is set for inst
//	if (warp_info.m_shader->get_sid() == 1 and m_warp_id == 0) {
//		printf("[%llu] Generating mem access for S1W0, space.get_type()=%d\n",
//			gpu_sim_cycle + gpu_tot_sim_cycle, space.get_type());
//	}

    if( empty() || op == MEMORY_BARRIER_OP || m_mem_accesses_created ) 
        return;
    if ( !((op == LOAD_OP) || (op == STORE_OP)) )
        return; 
    if( m_warp_active_mask.count() == 0 ) 
        return; // predicated off

    const size_t starting_queue_size = m_accessq.size();

    assert( is_load() || is_store() );
    assert( m_per_scalar_thread_valid ); // need address information per thread

    bool is_write = is_store();

    mem_access_type access_type;
    switch (space.get_type()) {
    case const_space:
    case param_space_kernel: 
        access_type = CONST_ACC_R; 
        break;
    case tex_space: 
        access_type = TEXTURE_ACC_R;   
        break;
    case global_space:       
        access_type = is_write? GLOBAL_ACC_W: GLOBAL_ACC_R;   
        break;
    case local_space:
    case param_space_local:  
        access_type = is_write? LOCAL_ACC_W: LOCAL_ACC_R;   
        break;
    case shared_space: break;
    default: {
    	printf("space.get_type() = %d\n", space.get_type());
    	assert(0); break;
    }
    }

    if (in_transaction) {
        // printf("TM Access Info: %s %d %d\n", m_tm_access_info.m_writelog_access.to_string().c_str(), 
        //        m_tm_access_info.m_conflict_detect, m_tm_access_info.m_version_managed);
    }

    // Calculate memory accesses generated by this warp
    new_addr_type cache_block_size = 0; // in bytes 

    switch( space.get_type() ) {
    case shared_space: {
        unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
        unsigned total_accesses=0;
        for( unsigned subwarp=0; subwarp <  m_config->mem_warp_parts; subwarp++ ) {

            // data structures used per part warp 
            std::map<unsigned,std::map<new_addr_type,unsigned> > bank_accs; // bank -> word address -> access count

            // step 1: compute accesses to words in banks
            for( unsigned thread=subwarp*subwarp_size; thread < (subwarp+1)*subwarp_size; thread++ ) {
                if( !active(thread) ) 
                    continue;
                new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
                //FIXME: deferred allocation of shared memory should not accumulate across kernel launches
                //assert( addr < m_config->gpgpu_shmem_size ); 
                unsigned bank = m_config->shmem_bank_func(addr);
                new_addr_type word = line_size_based_tag_func(addr,m_config->WORD_SIZE);
                bank_accs[bank][word]++;
            }

            if (m_config->shmem_limited_broadcast) {
                // step 2: look for and select a broadcast bank/word if one occurs
                bool broadcast_detected = false;
                new_addr_type broadcast_word=(new_addr_type)-1;
                unsigned broadcast_bank=(unsigned)-1;
                std::map<unsigned,std::map<new_addr_type,unsigned> >::iterator b;
                for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                    unsigned bank = b->first;
                    std::map<new_addr_type,unsigned> &access_set = b->second;
                    std::map<new_addr_type,unsigned>::iterator w;
                    for( w=access_set.begin(); w != access_set.end(); ++w ) {
                        if( w->second > 1 ) {
                            // found a broadcast
                            broadcast_detected=true;
                            broadcast_bank=bank;
                            broadcast_word=w->first;
                            break;
                        }
                    }
                    if( broadcast_detected ) 
                        break;
                }
            
                // step 3: figure out max bank accesses performed, taking account of broadcast case
                unsigned max_bank_accesses=0;
                for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                    unsigned bank_accesses=0;
                    std::map<new_addr_type,unsigned> &access_set = b->second;
                    std::map<new_addr_type,unsigned>::iterator w;
                    for( w=access_set.begin(); w != access_set.end(); ++w ) 
                        bank_accesses += w->second;
                    if( broadcast_detected && broadcast_bank == b->first ) {
                        for( w=access_set.begin(); w != access_set.end(); ++w ) {
                            if( w->first == broadcast_word ) {
                                unsigned n = w->second;
                                assert(n > 1); // or this wasn't a broadcast
                                assert(bank_accesses >= (n-1));
                                bank_accesses -= (n-1);
                                break;
                            }
                        }
                    }
                    if( bank_accesses > max_bank_accesses ) 
                        max_bank_accesses = bank_accesses;
                }

                // step 4: accumulate
                total_accesses+= max_bank_accesses;
            } else {
                // step 2: look for the bank with the maximum number of access to different words 
                unsigned max_bank_accesses=0;
                std::map<unsigned,std::map<new_addr_type,unsigned> >::iterator b;
                for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                    max_bank_accesses = std::max(max_bank_accesses, (unsigned)b->second.size());
                }

                // step 3: accumulate
                total_accesses+= max_bank_accesses;
            }
        }
        assert( total_accesses > 0 && total_accesses <= m_config->warp_size );
        cycles = total_accesses; // shared memory conflicts modeled as larger initiation interval 
        ptx_file_line_stats_add_smem_bank_conflict( pc, total_accesses );
        break;
    }

    case tex_space: 
        cache_block_size = m_config->gpgpu_cache_texl1_linesize;
        break;
    case const_space:  case param_space_kernel:
        cache_block_size = m_config->gpgpu_cache_constl1_linesize; 
        break;

    case global_space: case local_space: case param_space_local:
        if( space.get_type() == global_space and in_transaction and not m_config->no_tx_log_gen) {
            // detect cases where the access can be treated as normal, otherwise process it as transactional 
            bool generate_normal_accesses = false; 
            if (is_write) {
                if (m_tm_access_info.m_version_managed) {
                    generate_access_tm(is_write, access_type, warp_info); 
                } else {
                    generate_normal_accesses = true; 
                }
            } else {
                if (m_tm_access_info.m_conflict_detect) {
                    generate_access_tm(is_write, access_type, warp_info); 
                } else {
                    generate_normal_accesses = true; 
                }
            }
            if (generate_normal_accesses) {
                if(isatomic())
                    memory_coalescing_arch_13_atomic(is_write, access_type);
                else
                    memory_coalescing_arch_13(is_write, access_type);
            }
        } else if( m_config->gpgpu_coalesce_arch == 13 ) {
           if(isatomic())
               memory_coalescing_arch_13_atomic(is_write, access_type);
           else
               memory_coalescing_arch_13(is_write, access_type);
        } else abort();
        break;

    default:
        abort();
    }

    if( cache_block_size ) {
        assert( m_accessq.empty() );
        mem_access_byte_mask_t byte_mask; 
        std::map<new_addr_type,active_mask_t> accesses; // block address -> set of thread offsets in warp
        std::map<new_addr_type,active_mask_t>::iterator a;
        for( unsigned thread=0; thread < m_config->warp_size; thread++ ) {
            if( !active(thread) ) 
                continue;
            new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
            unsigned block_address = line_size_based_tag_func(addr,cache_block_size);
            accesses[block_address].set(thread);
            unsigned idx = addr-block_address; 
            for( unsigned i=0; i < data_size; i++ ) 
                byte_mask.set(idx+i);
        }
        for( a=accesses.begin(); a != accesses.end(); ++a ) 
            m_accessq.push_back( mem_access_t(access_type,a->first,cache_block_size,is_write,a->second,byte_mask) );
    }

    if ( space.get_type() == global_space ) {
        ptx_file_line_stats_add_uncoalesced_gmem( pc, m_accessq.size() - starting_queue_size );
    }
    m_mem_accesses_created=true;
}

#include "gpgpu-sim/shader.h"

unsigned g_debug_tm_write_log_entry_written = 0;

void tm_warp_info::backupRWSetToStack(simt_mask_t& staggered, ptx_thread_info** ptis, unsigned warp_id, int epoch, int sn)  {
   const bool dbg = (getenv("TOMMYDBG") != NULL);
   const unsigned rsize = m_read_log_size, wsize = m_write_log_size; // 0818
   const unsigned WARP_SIZE = 32;
   assert (staggered.size() == WARP_SIZE);


   bool should_print = false;
   shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(ptis[0]->get_core());
   if (scc && (scc->get_sid() == g_print_simt_stack_sid && warp_id == g_print_simt_stack_wid))
	   should_print = true;

   if (getenv("TOMMYDBG")) should_print = true;

   if (should_print) {
	   printf("[backupRWSetToStack-%llu] RS=%u=%u WS=%u=%u\n",
		  gpu_sim_cycle + gpu_tot_sim_cycle,
		  m_read_log_size, m_read_log.size(), m_write_log_size, m_write_log.size());
   }


   if (dbg) {
	   printf("[tm_warp_info::copyEntryToBackup] stagger: ");
	   for (unsigned i=0; i<staggered.size(); i++) {
		   if (staggered.test(i)) { printf("1"); } else { printf("0"); }
	   }
	   printf(", read_log_size: %u, write_log_size: %u\n", rsize, wsize);
   }

   backup_tx_acc_stack_entry_t bk_stk_ety;

   addr_t pc = (addr_t)(-1);
   for (int laneid = 0; laneid < WARP_SIZE; laneid++) {
	   if (staggered.test(laneid) == false) continue;
	   ptx_thread_info* pti = ptis[warp_id * 32 + laneid];
	   if (pti) {
		   pc = pti->get_pc();
		   break;
	   }
   }
   assert (pc != (addr_t)(-1));

   bool is_watched = false;
   for (int laneid = 0; laneid < WARP_SIZE; laneid ++) {
	   ptx_thread_info* pti = ptis[warp_id * 32 + laneid];
	   if (pti) {
		   if (pti->get_uid() == g_watched_tid) {
			   is_watched = true; break;
		   }
	   }
   }
   if (is_watched) {
	   printf("[backupRWSetToStack-%llu] Thd %u backup\n",
			   gpu_sim_cycle + gpu_tot_sim_cycle, g_watched_tid);
   }

   for (int laneid = 0; laneid < WARP_SIZE; laneid ++) {
//	   if (staggered.test(laneid) == false) continue;
	   if (dbg) printf("Lane %d:", laneid);
	   backup_tx_acc_per_thd_entry_t ety;

	   bool is_this_watched = false;
	   ptx_thread_info* pti = ptis[warp_id * 32 + laneid];
	   if (pti == NULL) continue;
	   if (pti->get_uid() == g_watched_tid) { is_this_watched = true; }

	   ety.lane_id = laneid;

	   // Usage:
	   //for (unsigned a = 0; a < m_read_log.size(); a++) {
	   //   fprintf(fout, "R[%2u] = [%#08x]%c\n", a, m_read_log[a].m_addr[lane_id], ((m_read_log[a].m_raw.test(a))? 'L':' '));
	   //}

	   if (is_this_watched) {
		   tm_manager* tm = dynamic_cast<tm_manager*>(pti->get_tm_manager());
		   if (tm) {
			   std::set<new_addr_type> words;
			   for (addr_set_t::iterator itr = tm->m_read_set.begin(); itr != tm->m_read_set.end(); itr++) {
			   	   words.insert((*itr) & 0xFFFFFFFC);
			   }
			   // LOG versus SET
			   printf("[backupRWSetToStack-%llu] Thd %u TM read word set (size=%lu):", gpu_sim_cycle + gpu_tot_sim_cycle, pti->get_uid(),
				   words.size());
			   for (std::set<new_addr_type>::iterator itr1 = words.begin(); itr1 != words.end(); itr1++) {
				   printf(" %x", *itr1);
			   }
			   printf("\n");
		   }
		   printf("[backupRWSetToStack-%llu] Thd %u backup Read LOG (size=%u):",
		       gpu_sim_cycle + gpu_tot_sim_cycle, pti->get_uid(), rsize);
	   }
	   for (unsigned idx_r = 0; idx_r < rsize; idx_r ++) {
		   ety.read_addrs.push_back(m_read_log[idx_r].m_addr[laneid]);
		   ety.read_raws.push_back(m_read_log[idx_r].m_raw[laneid]);
		   ety.read_actives.push_back(m_read_log[idx_r].m_active[laneid]);
		   if (is_this_watched) { printf(" %x", m_read_log[idx_r].m_addr[laneid]); }
	   }
	   if (is_this_watched) printf("\n");
	   for (unsigned idx_w = 0; idx_w < wsize; idx_w ++) {
		   ety.write_addrs.push_back(m_write_log[idx_w].m_addr[laneid]);
		   ety.write_raws.push_back(m_write_log[idx_w].m_raw[laneid]);
		   ety.write_actives.push_back(m_write_log[idx_w].m_active[laneid]);
		   if(dbg) printf(" W");
	   }

	   ptx_thread_info* pI = ptis[warp_id * 32 + laneid];
	   const ptx_thread_info& rpi = *pI;
	   ety.backup_pI = new ptx_thread_info(rpi);

	   for (std::list<ptx_thread_info::reg_map_t>::iterator itr = pI->m_regs.begin();
		   itr != pI->m_regs.end(); itr ++) {
		   ety.backup_pI->m_regs.push_back(*itr);
	   }

	   if(dbg) printf("\n");

	   bk_stk_ety.thds.push_back(ety);
   }
   bk_stk_ety.rsize = rsize; bk_stk_ety.wsize = wsize;
   bk_stk_ety.pc = pc;
   bk_stk_ety.epoch = epoch;
   bk_stk_ety.serial = sn;
   backup_stack.push_back(bk_stk_ety);
}

// Actually, this thing merges the R/W set into the current R/W set
int tm_warp_info::restoreRWSetFromStack(int idx, ptx_thread_info** ptis, unsigned warp_id, char merge_or_replace) {
   if (idx == -1) idx = backup_stack.size() - 1;

   addr_t entry_pc = 0xFFFFFFFF;
   bool should_print = false;
   shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(ptis[0]->get_core());
   if (scc && (scc->get_sid() == g_print_simt_stack_sid && warp_id == g_print_simt_stack_wid))
	   should_print = true;

   bool is_watched = false;
   ptx_thread_info* watched_pti = NULL;
   tm_manager* watched_tm = NULL;
   value_based_tm_manager* watched_vbtm = NULL;
   int watched_lane_id = -999;
   for (int i=0; i<32; i++) {
	   ptx_thread_info* pti = ptis[warp_id * 32 + i];
	   if (pti) {
		   if (pti->get_uid() == g_watched_tid) {
			   is_watched = true;
			   watched_pti = pti;
			   printf("Thread (%u,%u,%u)-(%u,%u,%u)\n",
			       pti->get_ctaid().x, pti->get_ctaid().y, pti->get_ctaid().z,
				   pti->get_tid().x,   pti->get_tid().y,   pti->get_tid().z   );
			   watched_tm = dynamic_cast<tm_manager*>(pti->get_tm_manager());
			   watched_vbtm = dynamic_cast<value_based_tm_manager*>(pti->get_tm_manager());
			   watched_lane_id = i;
			   break;
		   }
	   }
   }

   struct backup_tx_acc_stack_entry_t& stack_entry = backup_stack.at(idx);
   entry_pc = stack_entry.pc;
   int ret = stack_entry.serial;

   int saved_rsize = stack_entry.rsize, saved_wsize = stack_entry.wsize;
   int rsize = saved_rsize;
   int wsize = saved_wsize;
   if (merge_or_replace == 'M') {
	   rsize = (saved_rsize > m_read_log_size) ? saved_rsize : m_read_log_size;
	   wsize = (saved_wsize > m_write_log_size)? saved_wsize : m_write_log_size;
   }

   // Print.
   if (is_watched) {
	   printf("[restoreRWSetFromStack-%llu] Thd %u restore RW set, watched_tm=%p, merge_or_replace=%c, entry index=%d\n",
		   gpu_sim_cycle + gpu_tot_sim_cycle, g_watched_tid, watched_tm, merge_or_replace, idx);
	   if (watched_tm) {
		   std::set<new_addr_type> words;
		   for (addr_set_t::iterator itr = watched_tm->m_read_set.begin(); itr != watched_tm->m_read_set.end();
				itr ++) {
			   words.insert((*itr) & 0xFFFFFFFC);
		   }
		   printf("[restoreRWSetFromStack-%llu] Thd %u's tm_manager::m_read_set before restoration (size=%lu):",
				   gpu_sim_cycle + gpu_tot_sim_cycle, g_watched_tid, words.size());
		   for (std::set<new_addr_type>::iterator itr = words.begin(); itr != words.end(); itr++) {
			   printf(" %x", *itr);
		   }
		   printf("\n");
	   }
	   if (watched_vbtm) {
		   printf("[restoreRWSetFromStack-%llu] Thd %u's value_based_tm_manager::m_read_set_value before restoration (size=%lu):",
			   gpu_sim_cycle + gpu_tot_sim_cycle, g_watched_tid,
			   watched_vbtm->m_read_set_value.size());
		   for (tr1_hash_map<addr_t, unsigned int>::iterator itr = watched_vbtm->m_read_set_value.begin();
				itr != watched_vbtm->m_read_set_value.end(); itr++) {
			   printf(" %x", itr->first * 4);
		   }
		   printf("\n");
	   }

	   printf("[restoreRWSetFromStack] m_read_log before:\n");
	   for (int i=0; i<rsize; i++) {
		   printf("%x ", m_read_log[i].m_addr[watched_lane_id]);
	   }
	   printf("\n");
   }

   if (merge_or_replace == 'R') {
	   // The rsize and wsize are good, no need to change
	   m_read_log.clear();
	   m_write_log.clear();
	   for (unsigned x = 0; x < rsize; x++) m_read_log.push_back(tx_acc_entry_t(32));
	   for (unsigned x = 0; x < wsize; x++) m_write_log.push_back(tx_acc_entry_t(32));
   }

   m_read_log.resize(rsize);
   m_write_log.resize(wsize);

   for (unsigned lidx = 0; lidx < stack_entry.thds.size(); lidx ++) {
	   bool is_this_watched = false;
	   struct backup_tx_acc_per_thd_entry_t& thd_ety = stack_entry.thds.at(lidx);
	   unsigned lane_id = thd_ety.lane_id;
	   assert (thd_ety.read_actives.size() == saved_rsize && thd_ety.read_addrs.size() == saved_rsize &&
			   thd_ety.read_raws.size() == saved_rsize);
	   assert (thd_ety.write_actives.size() == saved_wsize && thd_ety.write_addrs.size() == saved_wsize &&
			   thd_ety.write_raws.size() == saved_wsize);
	   ptx_thread_info* pti = ptis[lane_id + 32 * warp_id];
	   if (pti->get_uid() == g_watched_tid) is_this_watched = true;


	   // 2015-09-21 hack
	   tm_manager* tmm = dynamic_cast<tm_manager*>(pti->get_tm_manager());
	   if (tmm) {
		   tmm->m_read_set.clear();
		   tmm->m_read_word_set.clear();
		   tmm->m_write_set.clear();
		   tmm->m_write_word_set.clear();
	   }
	   value_based_tm_manager* vbtm = dynamic_cast<value_based_tm_manager*>(pti->get_tm_manager());
	   value_based_tm_manager::addr_value_t old_rsvalue;
	   if (vbtm) {
		   old_rsvalue = vbtm->m_read_set_value;
		   if (is_this_watched) {
			   printf("[restoreRWSetFromStack] value_based_tm_manager::m_read_set_value has %lu entires before\n",
				  old_rsvalue.size());
		   }
		   vbtm->m_read_set_value.clear();
		   vbtm->m_read_word_set.clear();
	   }

	   std::set<new_addr_type> words1;
	   for (unsigned idx_r = 0; idx_r < saved_rsize; idx_r ++) {
		   addr_t word_addr = thd_ety.read_addrs.at(idx_r);
		   m_read_log[idx_r].m_addr[lane_id]   = thd_ety.read_addrs.at(idx_r);
		   m_read_log[idx_r].m_active[lane_id] = thd_ety.read_actives.at(idx_r);
		   m_read_log[idx_r].m_raw[lane_id]    = thd_ety.read_raws.at(idx_r);
		   words1.insert(thd_ety.read_addrs.at(idx_r) & 0xFFFFFFFC);

		   if (tmm) {
			   for (int i=0; i<4; i++) {
				   tmm->m_read_set.insert(word_addr + i);
			   }
			   tmm->m_read_word_set.insert(word_addr);
		   }
		   if (vbtm) {
			   for (int i=0; i<4; i++) {
				   addr_t addr = word_addr / 4;
				   if (old_rsvalue.find(addr) != old_rsvalue.end()) {
					   vbtm->m_read_set_value[addr] = old_rsvalue[addr];
				   }
			   }
		   }
	   }

	   if (is_this_watched) {
		   if (vbtm) {
			   printf("[restoreRWSetFromStack] value_based_tm_manager::m_read_set_value has %lu entires after\n",
				  vbtm->m_read_set_value.size());
		   }
		   printf("[restoreRWSetFromStack-%llu] Thd %u restored read log words (size=%lu):",
				gpu_sim_cycle + gpu_tot_sim_cycle, g_watched_tid, words1.size());
		   for (std::set<new_addr_type>::iterator itr = words1.begin(); itr != words1.end(); itr++) {
			   printf(" %x", *itr);
		   }
		   printf("\n");
	   }

	   for (unsigned idx_w = 0; idx_w < saved_wsize; idx_w ++) {
		   addr_t word_addr = thd_ety.write_addrs.at(idx_w);
		   m_write_log[idx_w].m_addr[lane_id]  = thd_ety.write_addrs.at(idx_w);
		   m_write_log[idx_w].m_active[lane_id]= thd_ety.write_actives.at(idx_w);
		   m_write_log[idx_w].m_raw[lane_id]   = thd_ety.write_raws.at(idx_w);
		   if (tmm) {
			   for (int i=0; i<4; i++) {
				   tmm->m_write_set.insert(word_addr + i);
			   }
			   tmm->m_write_word_set.insert(word_addr);
		   }
	   }

	   ptx_thread_info* pI = ptis[warp_id * 32 + lane_id];

// Ok.. what happens if we don't recover regs.

	   /*
	   pI->m_regs.clear();
	   pI->m_regs = std::list<ptx_thread_info::reg_map_t>();
	   for (std::list<ptx_thread_info::reg_map_t>::iterator itr = thd_ety.backup_pI->m_regs.begin();
			itr != thd_ety.backup_pI->m_regs.end(); itr ++) {
		   pI->m_regs.push_back( ptx_thread_info::reg_map_t() );
		   pI->m_regs.back() = *itr;
	   }*/
//	   pI->m_regs = thd_ety.backup_pI->m_regs;

//	   delete thd_ety.backup_pI;
   }

   backup_stack.erase(backup_stack.begin() + idx);

   m_read_log_size  = rsize; // ??????
   m_write_log_size = wsize; // ??????


   if (should_print) {
	   printf("[restoreRWSetFromStack-%llu] idx=%d RS=%u=%u WS=%u=%u PC=%x\n",
		  gpu_sim_cycle + gpu_tot_sim_cycle,
		  idx, m_read_log_size, m_read_log.size(), m_write_log_size, m_write_log.size(),
		  entry_pc);
   }


   if (is_watched) {
	   if (watched_tm) {
		   std::set<new_addr_type> words;
		   for (addr_set_t::iterator itr = watched_tm->m_read_set.begin(); itr != watched_tm->m_read_set.end();
				itr ++) {
			   words.insert((*itr) & 0xFFFFFFFC);
		   }
		   printf("[restoreRWSetFromStack-%llu] Thd %u's m_read_set after restoration (size=%lu):",
				   gpu_sim_cycle + gpu_tot_sim_cycle, g_watched_tid, words.size());
		   for (std::set<new_addr_type>::iterator itr = words.begin(); itr != words.end(); itr++) {
			   printf(" %x", *itr);
		   }
		   printf("\n");
	   }
	   if (watched_vbtm) {
		   printf("[restoreRWSetFromStack-%llu] Thd %u's value_based_tm_manager::m_read_set_value after restoration (size=%lu):\n",
			   gpu_sim_cycle + gpu_tot_sim_cycle, g_watched_tid,
			   watched_vbtm->m_read_set_value.size());
		   for (tr1_hash_map<addr_t, unsigned int>::iterator itr = watched_vbtm->m_read_set_value.begin();
				itr != watched_vbtm->m_read_set_value.end(); itr++) {
			   printf(" %x", itr->first * 4);
		   }
		   printf("\n");
	   }
	   printf("[restoreRWSetFromStack] m_read_log after:\n");
	   for (int i=0; i<rsize; i++) {
		   printf("%x ", m_read_log[i].m_addr[watched_lane_id]);
	   }
	   printf("\n");
   }

   return ret;
}

int tm_warp_info::restoreAllRWSetsFromStackByPCAndEpoch(int pc, ptx_thread_info** ptis, unsigned warp_id, int epoch_nolaterthan,
		std::vector<int>* sn) {
	int num_ety_used = 0;
	while (true) {
		bool is_found = false;
		int idx = 0;
		std::vector<struct backup_tx_acc_stack_entry_t>::iterator itr = backup_stack.begin();
		for (; itr != backup_stack.end(); itr++, idx++) {
			struct backup_tx_acc_stack_entry_t* ety = &(*itr);
			if (ety->pc == pc && ety->epoch <= epoch_nolaterthan) {
				is_found = true;
				break;
			}
		}
		if (!is_found) break;
		else {
			int one_sn = restoreRWSetFromStack(idx, ptis, warp_id, 'M');
			sn->push_back(one_sn);
			num_ety_used ++;
		}
	}
	return num_ety_used;
}

void tm_warp_info::reset()
{
   if (getenv("TOMMYDBG")) {
	   printf("[tm_warp_info::reset] Cycle=%llu \n", gpu_sim_cycle + gpu_tot_sim_cycle);
   }
   m_read_log_size = 0;
   m_write_log_size = 0;
   m_read_log.clear();
   m_write_log.clear();
}

void tm_warp_info::print_log(unsigned int lane_id, FILE *fout) 
{
   fprintf(fout, "R size=%u=%u, W size=%u=%u\n", m_read_log_size, m_read_log.size(), m_write_log_size, m_write_log.size());
   for (unsigned a = 0; a < m_read_log.size(); a++) {
      fprintf(fout, "R[%2u] = [%#08x]%c\n", a, m_read_log[a].m_addr[lane_id], ((m_read_log[a].m_raw.test(a))? 'L':' '));
   }
   for (unsigned a = 0; a < m_write_log.size(); a++) {
      fprintf(fout, "W[%2u] = [%#08x]%c\n", a, m_write_log[a].m_addr[lane_id], ((m_write_log[a].m_raw.test(a))? 'L':' '));
   }
}

void tm_warp_info::reset_lane(unsigned int lane_id)
{
   for (unsigned a = 0; a < m_read_log.size(); a++) {
      m_read_log[a].m_addr[lane_id] = 0;
      m_read_log[a].m_raw.reset(lane_id); 
      m_read_log[a].m_active.reset(lane_id); 
   }
   
   for (unsigned a = 0; a < m_write_log.size(); a++) {
      m_write_log[a].m_addr[lane_id] = 0;
      m_write_log[a].m_raw.reset(lane_id); 
      m_write_log[a].m_active.reset(lane_id); 
   }
}

// return the number of access generated for filling the transaction read log 
unsigned warp_inst_t::n_txlog_fill() const 
{
    if (not in_transaction) 
        assert (m_txlog_fill_accesses == 0); 
    return m_txlog_fill_accesses; 
}

// add a entry to write/read log visible to uarch model 
void warp_inst_t::append_log(tm_warp_info::tx_log_t &log, addr_t offset, unsigned warp_size, std::bitset<32> &raw_access)
{
   tm_warp_info::tx_acc_entry_t tx_entry(warp_size); 
   for (unsigned t = 0; t < warp_size; t++) {
      if (active(t)) {
         tx_entry.m_addr[t] = m_per_scalar_thread[t].memreqaddr[0] + offset; // Assuming 4B chunk for access log
      } else {
         tx_entry.m_addr[t] = 0;
      }
   }
   tx_entry.m_raw = raw_access; // save the writelog access vector
   tx_entry.m_active = m_warp_active_mask; 
   log.push_back(tx_entry);
}

addr_t tm_warp_info::write_log_offset = /*8192*/ 300*1024; // WF: Is this large enough? // To run RBTree + both, set to 30*1024. Don't know why it causes crashes
addr_t tm_warp_info::read_log_offset = /*4096*/ 4096;

void warp_inst_t::generate_access_tm( bool is_write, mem_access_type access_type, tm_warp_info &warp_info )
{
   const unsigned addr_size = 4; 
   const unsigned word_size = 4; 
   const active_mask_t empty_mask;
   mem_access_byte_mask_t full_byte_mask; 
   full_byte_mask.set(); 
   unsigned access_data_size = data_size * ldst_vector_elm; 
   m_txlog_fill_accesses = 0;
   if ( is_write ) {
      assert(m_tm_access_info.m_version_managed == true);
      unsigned wtid = m_warp_id * m_config->warp_size; 
      unsigned atag_block_size = addr_size * m_config->warp_size; 
      unsigned data_block_size = word_size * m_config->warp_size; 
      for (unsigned w = 0; w * word_size < access_data_size; w++) {
         // for each word written, generate two stores <Address, Value> to local memory 
         unsigned int next_write_entry = warp_info.m_write_log_size++; 
         addr_t next_data_block = 
            warp_info.m_shader->translate_local_memaddr((next_write_entry*2+1) * word_size + tm_warp_info::write_log_offset,
                                                        wtid, word_size); 
         m_accessq.push_back( mem_access_t(LOCAL_ACC_W,next_data_block,data_block_size,true,m_warp_active_mask,full_byte_mask,
        		 this->m_serial_number) );
         addr_t next_atag_block = 
            warp_info.m_shader->translate_local_memaddr(next_write_entry*2 * word_size + tm_warp_info::write_log_offset,
                                                        wtid, addr_size); 
         m_accessq.push_back( mem_access_t(LOCAL_ACC_W,next_atag_block,atag_block_size,true,empty_mask,full_byte_mask,
        		 this->m_serial_number) );

         // printf("tm_store: wtlog=%d atag_block=%#08x data_block=%#08x m_accessqsize=%zd\n", next_write_entry, next_atag_block, next_data_block, m_accessq.size());
         g_debug_tm_write_log_entry_written += 2; 
         
         // add a entry to write log visible to uarch model 
         append_log(warp_info.m_write_log, w * word_size, m_config->warp_size, m_tm_access_info.m_writelog_access);

         assert (warp_info.m_write_log.size() == warp_info.m_write_log_size); // TOMMY0812

         #if 0
         if (warp_info.m_shader->get_sid() == 18 and m_warp_id == 14) {
            printf("activemask = %08x\n", m_warp_active_mask.to_ulong()); 
            printf("timeout_validation_fail = %08x\n", m_tm_access_info.m_timeout_validation_fail.to_ulong()); 
            for (unsigned t = 0; t < m_config->warp_size; t++) {
               printf("lane %u\n", t); 
               warp_info.print_log(t, stdout); 
            }
         }
         #endif
      }
   } else {
      assert(m_tm_access_info.m_conflict_detect == true);
      // detect RAW access and walk write log if needed, remove those RAW access from the memory coalescing
      active_mask_t original_active_mask = m_warp_active_mask; 
      if (m_tm_access_info.m_writelog_access.any()) {
    	  if (!(warp_info.m_write_log_size > 0)) {
    		  printf("S%uW%u warp_info.m_write_log_size = %u cycle=%llu\n",
				  warp_info.m_shader->get_sid(), m_warp_id,
				  warp_info.m_write_log_size,
				  gpu_sim_cycle + gpu_tot_sim_cycle);
    		  printf("m_tm_access_info.m_writelog_access is: ");
    		  for (unsigned i=0; i<m_tm_access_info.m_writelog_access.size(); i++) {
    			  if (m_tm_access_info.m_writelog_access.test(i)) printf("1"); else printf("0");
    		  }
    		  printf("\n");
    	  }
    	  assert(warp_info.m_write_log_size > 0);
      }
      m_warp_active_mask &= ~m_tm_access_info.m_writelog_access; // clear bit for any thread that have RAW hit

      #if 0
      if (warp_info.m_shader->get_sid() == 1 and m_warp_id == 0) {
         printf("access_data_size = %u\n", access_data_size);
         for (unsigned t = 0; t < m_config->warp_size; t++) {
            if (active(t)) {
               printf("t%02u ", t); 
               for (unsigned x = 0; x < 4; x++) 
                  printf("[%#08x] ", (unsigned int)m_per_scalar_thread[t].memreqaddr[x]); 
               printf("\n"); 
            } else {
               printf("t%02u \n", t); 
            }
         }
      }
      #endif

      // generate the actual memory fetch
      if(isatomic())
          memory_coalescing_arch_13_atomic(is_write, access_type);
      else
          memory_coalescing_arch_13(is_write, access_type);

      // change the fill address (but not the fetch address) to read log in local memory 
      unsigned wtid = m_warp_id * m_config->warp_size; 
      unsigned int read_entry = warp_info.m_read_log_size; 
      addr_t data_block[4];  // each instruction may load up to 128-bits, writing back the data fields in up to 4 entries 
      for (unsigned w = 0; w * word_size < access_data_size; w++) {
         data_block[w] = 
            warp_info.m_shader->translate_local_memaddr(((read_entry+w)*2+1) * word_size + tm_warp_info::read_log_offset,
                                                        wtid, word_size);
         assert (((read_entry+w)*2+1)*word_size < (tm_warp_info::write_log_offset-tm_warp_info::read_log_offset)
        		 && "Read log overflow!");
      }
      for (std::list<mem_access_t>::iterator iAcc = m_accessq.begin(); iAcc != m_accessq.end(); ++iAcc) {
         if (iAcc->get_type() == GLOBAL_ACC_R) {
            iAcc->set_tx_load(data_block[0]); // just set them all to write to data block 0
         }
      }

      unsigned atag_block_size = addr_size * m_config->warp_size; 
      unsigned data_block_size = word_size * m_config->warp_size;

      if(m_warp_active_mask.any()){
          // HACK: for data blocks in other entries, just create a store to make sure the entry is allocated in cache 
          for (unsigned w = 1; w * word_size < access_data_size; w++) {
             m_accessq.push_back( mem_access_t(LOCAL_ACC_W,data_block[w],data_block_size,true,empty_mask,full_byte_mask,
            		 this->m_serial_number) );
             m_txlog_fill_accesses += 1; // tell the ldst_unit to not treat these accesses as pending writebacks 
             // printf("tm_load: rdlog=%d data_block=%#08x m_accessqsize=%zd\n", read_entry + w, data_block[w], m_accessq.size());
          }

          for (unsigned w = 0; w * word_size < access_data_size; w++) {
             // for each word read, write address tags to local memory
             unsigned int next_read_entry = warp_info.m_read_log_size++;
             addr_t next_atag_block =
                warp_info.m_shader->translate_local_memaddr(next_read_entry*2 * word_size + tm_warp_info::read_log_offset,
                                                            wtid, addr_size);
             assert ((next_read_entry*2)*word_size < (tm_warp_info::write_log_offset-tm_warp_info::read_log_offset)
                     		 && "Read log overflow!");
             m_accessq.push_back( mem_access_t(LOCAL_ACC_W,next_atag_block,atag_block_size,true,empty_mask,full_byte_mask,
            		 this->m_serial_number) );
             m_txlog_fill_accesses += 1; // tell the ldst_unit to not treat these accesses as pending writebacks 

             // printf("tm_load: rdlog=%d atag_block=%#08x data_block=%#08x m_accessqsize=%zd\n", next_read_entry, next_atag_block, next_data_block, m_accessq.size());

             // add a entry to read log visible to uarch model
             unsigned old_size = warp_info.m_read_log.size();
             append_log(warp_info.m_read_log, w * word_size, m_config->warp_size, m_tm_access_info.m_writelog_access);

             if (! (warp_info.m_read_log.size() == warp_info.m_read_log_size)) { // TOMMY0812
            	 printf("read_log_size() vs size is %lu vs %u; old_size = %u\n",
            	     warp_info.m_read_log.size(), warp_info.m_read_log_size, old_size);
            	 assert(0);
             }

             #if 0
             if (warp_info.m_shader->get_sid() == 18 and m_warp_id == 14) {
                printf("activemask = %08x\n", m_warp_active_mask.to_ulong()); 
                printf("timeout_validation_fail = %08x\n", m_tm_access_info.m_timeout_validation_fail.to_ulong()); 
                for (unsigned t = 0; t < m_config->warp_size; t++) {
                   printf("lane %u\n", t); 
                   warp_info.print_log(t, stdout); 
                }
             }
             #endif
          }
      }

      if (m_tm_access_info.m_writelog_access.any()) {
         // generate write log walk in running order (the stack will reverse the order -- the intended order)
         // assuming worst case traversal through the whole log
         for (unsigned w = 0; w < warp_info.m_write_log_size; w++) {
            if (w == 0) {
               // only one word will hit -- generate a load <data> from local memory 
               addr_t next_data_block = 
                  warp_info.m_shader->translate_local_memaddr((w*2+1) * word_size + tm_warp_info::write_log_offset,
                                                              wtid, word_size); 
               m_accessq.push_back( mem_access_t(LOCAL_ACC_R,next_data_block,data_block_size,false,empty_mask,full_byte_mask) );
            }
            // for each write log entry, generate a load <Address> from local memory 
            addr_t next_atag_block = 
               warp_info.m_shader->translate_local_memaddr(w*2 * word_size + tm_warp_info::write_log_offset,
                                                           wtid, addr_size); 
            m_accessq.push_back( mem_access_t(LOCAL_ACC_R,next_atag_block,atag_block_size,false,empty_mask,full_byte_mask) );

            // printf("tm_load: wtlog=%d atag_block=%#08x data_block=%#08x m_accessqsize=%zd\n", next_write_entry, next_atag_block, next_data_block, m_accessq.size());
         }
      }

      m_warp_active_mask = original_active_mask; // restore active mask 
   }
}

void warp_inst_t::memory_coalescing_arch_13( bool is_write, mem_access_type access_type )
{
    // see the CUDA manual where it discusses coalescing rules before reading this
    unsigned segment_size = 0;
    unsigned warp_parts = m_config->mem_warp_parts;
    switch( data_size ) {
    case 1: segment_size = 32; break;
    case 2: segment_size = 64; break;
    case 4: case 8: case 16: segment_size = 128; break;
    }
    unsigned subwarp_size = m_config->warp_size / warp_parts;

    for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
        std::map<new_addr_type,transaction_info> subwarp_transactions;

        // step 1: find all transactions generated by this subwarp
        for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
            if( !active(thread) )
                continue;

            unsigned data_size_coales = data_size;
            unsigned num_accesses = 1;

            if( space.get_type() == local_space || space.get_type() == param_space_local ) {
               // Local memory accesses >4B were split into 4B chunks
               if(data_size >= 4) {
                  data_size_coales = 4;
                  num_accesses = data_size/4;
               }
               // Otherwise keep the same data_size for sub-4B access to local memory
            }


            assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);

            for(unsigned access=0; access<num_accesses; access++) {
                new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[access];
                unsigned block_address = line_size_based_tag_func(addr,segment_size);
                unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?
                transaction_info &info = subwarp_transactions[block_address];

                // can only write to one segment
                assert(block_address == line_size_based_tag_func(addr+data_size_coales-1,segment_size));

                info.chunks.set(chunk);
                info.active.set(thread);
                unsigned idx = (addr&127);
                for( unsigned i=0; i < data_size_coales; i++ )
                    info.bytes.set(idx+i);
            }
        }

        // step 2: reduce each transaction size, if possible
        std::map< new_addr_type, transaction_info >::iterator t;
        for( t=subwarp_transactions.begin(); t !=subwarp_transactions.end(); t++ ) {
            new_addr_type addr = t->first;
            const transaction_info &info = t->second;

            memory_coalescing_arch_13_reduce_and_send(is_write, access_type, info, addr, segment_size);

        }
    }
}

void warp_inst_t::memory_coalescing_arch_13_atomic( bool is_write, mem_access_type access_type )
{

   assert(space.get_type() == global_space); // Atomics allowed only for global memory

   // see the CUDA manual where it discusses coalescing rules before reading this
   unsigned segment_size = 0;
   unsigned warp_parts = 2;
   switch( data_size ) {
   case 1: segment_size = 32; break;
   case 2: segment_size = 64; break;
   case 4: case 8: case 16: segment_size = 128; break;
   }
   unsigned subwarp_size = m_config->warp_size / warp_parts;

   for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
       std::map<new_addr_type,std::list<transaction_info> > subwarp_transactions; // each block addr maps to a list of transactions

       // step 1: find all transactions generated by this subwarp
       for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
           if( !active(thread) )
               continue;

           new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
           unsigned block_address = line_size_based_tag_func(addr,segment_size);
           unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?

           // can only write to one segment
           assert(block_address == line_size_based_tag_func(addr+data_size-1,segment_size));

           // Find a transaction that does not conflict with this thread's accesses
           bool new_transaction = true;
           std::list<transaction_info>::iterator it;
           transaction_info* info;
           for(it=subwarp_transactions[block_address].begin(); it!=subwarp_transactions[block_address].end(); it++) {
              unsigned idx = (addr&127);
              if(not it->test_bytes(idx,idx+data_size-1)) {
                 new_transaction = false;
                 info = &(*it);
                 break;
              }
           }
           if(new_transaction) {
              // Need a new transaction
              subwarp_transactions[block_address].push_back(transaction_info());
              info = &subwarp_transactions[block_address].back();
           }
           assert(info);

           info->chunks.set(chunk);
           info->active.set(thread);
           unsigned idx = (addr&127);
           for( unsigned i=0; i < data_size; i++ ) {
               assert(!info->bytes.test(idx+i));
               info->bytes.set(idx+i);
           }
       }

       // step 2: reduce each transaction size, if possible
       std::map< new_addr_type, std::list<transaction_info> >::iterator t_list;
       for( t_list=subwarp_transactions.begin(); t_list !=subwarp_transactions.end(); t_list++ ) {
           // For each block addr
           new_addr_type addr = t_list->first;
           const std::list<transaction_info>& transaction_list = t_list->second;

           std::list<transaction_info>::const_iterator t;
           for(t=transaction_list.begin(); t!=transaction_list.end(); t++) {
               // For each transaction
               const transaction_info &info = *t;
               memory_coalescing_arch_13_reduce_and_send(is_write, access_type, info, addr, segment_size);
           }
       }
   }
}

void warp_inst_t::memory_coalescing_arch_13_reduce_and_send( bool is_write, mem_access_type access_type, const transaction_info &info, new_addr_type addr, unsigned segment_size )
{
   assert( (addr & (segment_size-1)) == 0 );

   const std::bitset<4> &q = info.chunks;
   assert( q.count() >= 1 );
   std::bitset<2> h; // halves (used to check if 64 byte segment can be compressed into a single 32 byte segment)

   unsigned size=segment_size;
   if( segment_size == 128 ) {
       bool lower_half_used = q[0] || q[1];
       bool upper_half_used = q[2] || q[3];
       if( lower_half_used && !upper_half_used ) {
           // only lower 64 bytes used
           size = 64;
           if(q[0]) h.set(0);
           if(q[1]) h.set(1);
       } else if ( (!lower_half_used) && upper_half_used ) {
           // only upper 64 bytes used
           addr = addr+64;
           size = 64;
           if(q[2]) h.set(0);
           if(q[3]) h.set(1);
       } else {
           assert(lower_half_used && upper_half_used);
       }
   } else if( segment_size == 64 ) {
       // need to set halves
       if( (addr % 128) == 0 ) {
           if(q[0]) h.set(0);
           if(q[1]) h.set(1);
       } else {
           assert( (addr % 128) == 64 );
           if(q[2]) h.set(0);
           if(q[3]) h.set(1);
       }
   }
   if( size == 64 ) {
       bool lower_half_used = h[0];
       bool upper_half_used = h[1];
       if( lower_half_used && !upper_half_used ) {
           size = 32;
       } else if ( (!lower_half_used) && upper_half_used ) {
           addr = addr+32;
           size = 32;
       } else {
           assert(lower_half_used && upper_half_used);
       }
   }
   m_accessq.push_back( mem_access_t(access_type,addr,size,is_write,info.active,info.bytes,
	   this->m_serial_number) );
}

void warp_inst_t::completed( unsigned long long cycle ) const 
{
   unsigned long long latency = cycle - issue_cycle; 
   assert(latency <= cycle); // underflow detection 
   ptx_file_line_stats_add_latency(pc, latency * active_count());  
}


unsigned kernel_info_t::m_next_uid = 1;

kernel_info_t::kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry )
{
    m_kernel_entry=entry;
    m_grid_dim=gridDim;
    m_block_dim=blockDim;
    m_next_cta.x=0;
    m_next_cta.y=0;
    m_next_cta.z=0;
    m_next_tid=m_next_cta;
    m_num_cores_running=0;
    m_uid = m_next_uid++;
    m_param_mem = new memory_space_impl<8192>("param",64*1024);
}

kernel_info_t::~kernel_info_t()
{
    assert( m_active_threads.empty() );
    delete m_param_mem;
}

std::string kernel_info_t::name() const
{
    return m_kernel_entry->get_name();
}

simt_stack::simt_stack( unsigned wid, unsigned warpSize, core_t* core)
{
    m_warp_id=wid;
    m_warp_size = warpSize;
    m_core = core;
    m_epoch = 0;
    reset();
}

void simt_stack::reset()
{
    m_stack.clear();
    m_in_transaction = false; 
}

void simt_stack::launch( address_type start_pc, const simt_mask_t &active_mask )
{
    reset();
    simt_stack_entry new_stack_entry(0);
    new_stack_entry.m_pc = start_pc;
    new_stack_entry.m_calldepth = 1;
    new_stack_entry.m_active_mask = active_mask;
    new_stack_entry.m_type = STACK_ENTRY_TYPE_NORMAL;
    m_stack.push_back(new_stack_entry);

    CartographerTimeSeries_onWarpActiveMaskChanged(m_core, this->m_warp_id, &active_mask, "Launch");
}

const simt_mask_t &simt_stack::get_active_mask() const
{
    assert(m_stack.size() > 0);
    return m_stack.back().m_active_mask;
}

const enum simt_stack::stack_entry_type &simt_stack::get_type() const
{
    assert(m_stack.size() > 0);
    return m_stack.back().m_type;
}

void simt_stack::get_pdom_stack_top_info( unsigned *pc, unsigned *rpc ) const
{
   assert(m_stack.size() > 0);
   *pc = m_stack.back().m_pc;
   *rpc = m_stack.back().m_recvg_pc;
}

unsigned simt_stack::get_rp() const 
{ 
    assert(m_stack.size() > 0);
    return m_stack.back().m_recvg_pc;
}

void simt_stack::print (FILE *fout) const
{
    for ( unsigned k=0; k < m_stack.size(); k++ ) {
        simt_stack_entry stack_entry = m_stack[k];
        if ( k==0 ) {
            fprintf(fout, "w%02d %1u ", m_warp_id, k );
        } else {
            fprintf(fout, "    %1u ", k );
        }
        char c_type = '?'; 
        switch (stack_entry.m_type) {
            case STACK_ENTRY_TYPE_INVALID: c_type = 'I'; break; 
            case  STACK_ENTRY_TYPE_NORMAL: c_type = 'N'; break; 
            case   STACK_ENTRY_TYPE_RETRY: c_type = 'R'; break; 
            case   STACK_ENTRY_TYPE_TRANS: c_type = 'T'; break; 
            case    STACK_ENTRY_TYPE_CALL: c_type = 'C'; break; 
            default: c_type = '?'; break; 
        };
        for (unsigned j=0; j<m_warp_size; j++)
            fprintf(fout, "%c", (stack_entry.m_active_mask.test(j)?'1':'0') );
        fprintf(fout, " pc: 0x%03x", stack_entry.m_pc );
        if ( stack_entry.m_recvg_pc == (unsigned)-1 ) {
            fprintf(fout," rp: ---- tp: %c cd: %2u ", c_type, stack_entry.m_calldepth );
        } else {
            fprintf(fout," rp: 0x%03x tp: %c cd: %2u ", stack_entry.m_recvg_pc, c_type, stack_entry.m_calldepth );
        }
        if ( stack_entry.m_branch_div_cycle != 0 ) {
            fprintf(fout," bd@%6u ", (unsigned) stack_entry.m_branch_div_cycle );
        } else {
            fprintf(fout," " );
        }
        ptx_print_insn( stack_entry.m_pc, fout );
        fprintf(fout,"\n");
    }
}

void simt_stack::sanityCheck0830() {
	unsigned sid = (unsigned)-1, wid = m_warp_id;
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
	if (scc) sid = scc->get_sid();

	// Assertion: Bit-wise add R entry through the TOS, every bit should be <= 1.
	g_0830_sanity_check_count ++;
	int index_r = -999, num_r = 0, index_last_t = -999;
	for (int i=0; i<m_stack.size(); i++) {
		if (m_stack[i].m_type == STACK_ENTRY_TYPE_RETRY) {
			index_r = i; num_r ++;
		}
		if (m_stack[i].m_type == STACK_ENTRY_TYPE_TRANS) {
			index_last_t = i; g_0830_t_count ++;
		}
	}

	if ((index_last_t != (m_stack.size() - 1)) && (index_last_t != -999)) {
		for (int i=0; i<m_stack.size(); i++) {
			if (m_stack[i].m_type == STACK_ENTRY_TYPE_NORMAL)
				g_0830_n_count ++;
		}
	}

	if (index_r == -999) return;
	if (num_r > 1) assert(false);
	for (int i=index_r; i<=index_last_t; i++) {
		if (m_stack[i].m_type == STACK_ENTRY_TYPE_NORMAL) {
			this->print(stdout);
			printf("Oh!\n");
			assert(0);
		}
	}
	active_mask_t sum_mask;
	for (int i=index_r; i<=index_last_t; i++) {
		active_mask_t curr = m_stack[i].m_active_mask;
		for (int i=0; i<sum_mask.size(); i++) {
			if (curr.test(i)) {
				if (sum_mask.test(i)) {
					this->print(stdout);
					printf("Oh!!! Assertion Failed !!!!!!!!!!!!\n");
					assert(false);
				}
			}
		}
		sum_mask |= curr;
	}

	if (index_last_t == -999) return;
	active_mask_t last_t_mask = m_stack[index_last_t].m_active_mask;
	sum_mask.reset();
	for (int i=1+index_last_t; i<m_stack.size(); i++) {
		if (!(m_stack[i].m_type == STACK_ENTRY_TYPE_NORMAL)) {
			printf("Oh! There is a non-N entry on top of the top T entry.\n");
			this->print(stdout);
			assert(0);
		}

		active_mask_t curr = m_stack[i].m_active_mask;
		for (int i=0; i<curr.size(); i++) {
			if (curr.test(i)) {
				if (sum_mask.test(i)) {
					printf("\n");
					this->print(stdout);
					printf("S%uW%u Oh! Sanity check error 2\n", sid, wid);
					assert(false);
				}
			}
		}
		sum_mask |= curr;
	}

	for (int i=0; i<sum_mask.size(); i++) {
		if (sum_mask.test(i)) {
			if (not last_t_mask.test(i)) {
				printf ("Oh! Sanity check error 3\n");
				assert (false);
			}
		}
	}
}

void simt_stack::update( simt_mask_t &thread_done, simt_mask_t thread_stagger, addr_vector_t &next_pc, address_type recvg_pc, op_type next_inst_op, address_type curr_pc )
{
	bool break_here = false;
	if (gpu_sim_cycle + gpu_tot_sim_cycle >= g_tommy_break_cycle) {
		break_here = true;
	}

	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
	int sid = -1, wid = m_warp_id;
	if (scc) { sid = scc->m_sid; }
	bool should_print = false;
	if (g_print_simt_stack_sid == sid && g_print_simt_stack_wid == wid && g_tommy_break_cycle <= gpu_sim_cycle + gpu_tot_sim_cycle)
		should_print = true;


	if (next_inst_op == TRIPWIRE_OP) {
		printf("[simt_stack::update] W%u Saw a TRIPWIRE_OP, current stack:\n", m_warp_id);
		this->print(stdout);
		printf("----------------------------------------\n");
	}

    assert(m_stack.size() > 0);

    assert( next_pc.size() == m_warp_size );

    // TOMMY GUN
    if (m_stack.back().m_type == STACK_ENTRY_TYPE_RETRY && m_stack.back().m_active_mask.none()) {
    	m_stack.pop_back();
    	address_type pc1 = m_stack.back().m_pc;
    	for (int i=0; i<next_pc.size(); i++) next_pc[i] = pc1;
    	if (scc) {
			for (unsigned t=0; t<m_warp_size; t++) {
				unsigned tid = t + m_warp_size * m_warp_id;
				ptx_thread_info* pti = scc->m_thread[tid];
				if (pti) pti->m_PC = pti->m_NPC = pc1;
			}
    	}
    }

    simt_mask_t  top_active_mask = m_stack.back().m_active_mask;
    address_type top_recvg_pc = m_stack.back().m_recvg_pc;
    address_type top_pc = m_stack.back().m_pc; // the pc of the instruction just executed
    stack_entry_type top_type = m_stack.back().m_type;
    int top_epoch = m_stack.back().m_epoch;
    unsigned long long top_branch_div_cyc = m_stack.back().m_branch_div_cycle;

    // pop all invalid entries from the stack (due to TX-abort of non-TOS threads)
    while (top_type == STACK_ENTRY_TYPE_INVALID) {
        assert(m_stack.back().m_active_mask.none()); 
        m_stack.pop_back(); 
        top_active_mask = m_stack.back().m_active_mask;
        top_recvg_pc = m_stack.back().m_recvg_pc;
        top_pc = m_stack.back().m_pc; 
        top_type = m_stack.back().m_type;
        top_branch_div_cyc = m_stack.back().m_branch_div_cycle;
        top_epoch = m_stack.back().m_epoch;
    }

    if (should_print) {

    	printf("S%uW%u curr_pc=%x\n", sid, wid, curr_pc);
    	printf("Next PCs:");
    	for (addr_vector_t::iterator itr = next_pc.begin(); itr != next_pc.end(); itr++)
    		printf(" %x", *itr);
    	printf("\n");
    	this->print(stdout);
    	printf("Stagger: ");
    	for (unsigned i=0; i<thread_stagger.size(); i++) {
    		if (thread_stagger.test(i)) printf("1"); else printf("0");
    	}
    	printf("\n");
    }

    if (!top_active_mask.any()) {
    	printf("S%uW%u curr_pc=%x cycle=%llu\n", sid, wid, curr_pc, gpu_sim_cycle + gpu_tot_sim_cycle);
    	this->print(stdout);
    	assert(0);
    }

    const address_type null_pc = -1;
    bool warp_diverged = false;
    address_type new_recvg_pc = null_pc;

    if (thread_stagger.any()) {
    	bool should_skip = false;
    	// TODO: For debugging 0818

    	address_type my_cp = get_my_converge_point(curr_pc);
    	if (my_cp == (address_type)(-2)) my_cp = top_recvg_pc;

    	if (m_stack.back().m_type != STACK_ENTRY_TYPE_TRANS) {
			{
			    if (getenv("TOMMYDBG")) {
					printf("If we use reconverge point mode 1 and 0, we cannot do a stagger inside a branch.");
					this->print(stdout);
					assert (0);
				} else {
					should_skip = true;
					thread_stagger.reset();
				}
			}
    	}

		if (sid == g_print_simt_stack_sid && wid == g_print_simt_stack_wid) {
			printf("[simt_stack-%llu] I have some stagger.\n", gpu_sim_cycle + gpu_tot_sim_cycle);

			printf("Next_PC's:");
			for (unsigned i=0; i<next_pc.size(); i++) {
				printf(" %x", next_pc.at(i));
			}
			printf(", thread_stagger: ");
			for (unsigned i=0; i<thread_stagger.size(); i++) {
				if (thread_stagger.test(i)) printf("1");
				else printf("0");
			}
			printf("\n");

			printf("Current stack (before stagger):\n");
			this->print(stdout);
		}
    	if (should_skip == false) {


			simt_mask_t stagger_mask;
			stagger_mask.reset();
			for (unsigned i=0; i<thread_stagger.size(); i++) {
				if (thread_stagger.test(i)) {
					stagger_mask.set(i);
					assert (next_pc[i] == curr_pc); // Set in functional simulation.
					top_active_mask.reset(i);
				}
			}

			assert (curr_pc != 0xffffffff);

			simt_stack_entry stagger_ety(top_epoch);
			stagger_ety.m_pc = curr_pc;
			stagger_ety.m_active_mask = stagger_mask;
			stagger_ety.m_type = STACK_ENTRY_TYPE_TRANS;
			stagger_ety.m_recvg_pc = 0xFFFFFFFF;
			stagger_ety.m_branch_div_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;

			simt_stack_entry old_top_ety = m_stack.back();
			m_stack.pop_back();
			m_stack.push_back(stagger_ety);
			m_stack.push_back(old_top_ety);
			m_stack.back().m_active_mask = top_active_mask;

			if (scc) {
				tm_warp_info& twi = scc->m_warp[m_warp_id].get_tm_warp_info();
				twi.backupRWSetToStack(stagger_mask, scc->m_thread, m_warp_id, m_epoch, m_stack.back().m_serial);
			}

			if (should_print) {
				printf("Stack-after inserted stagger_ety:\n");
				this->print(stdout);
			}
    	}
    }

    while (top_active_mask.any()) {

        // extract a group of threads with the same next PC among the active threads in the warp
        address_type tmp_next_pc = null_pc;
        simt_mask_t tmp_active_mask;
        for (int i = m_warp_size - 1; i >= 0; i--) {
            if ( top_active_mask.test(i) ) { // is this thread active?
                if (thread_done.test(i)) {
                    top_active_mask.reset(i); // remove completed thread from active mask
                } else if (tmp_next_pc == null_pc) {
                    tmp_next_pc = next_pc[i];
                    tmp_active_mask.set(i);
                    top_active_mask.reset(i);
                } else if (tmp_next_pc == next_pc[i]) {
                    tmp_active_mask.set(i);
                    top_active_mask.reset(i);
                }
            }
        }

        if(tmp_next_pc == null_pc) {
            assert(!top_active_mask.any()); // all threads done
            continue;
        }

        if (next_inst_op == TRIPWIRE_OP && getenv("IGNORE_TRIPWIRE")==NULL) {
        	assert (top_active_mask.any() == false);
        	address_type my_cp = get_my_converge_point(curr_pc);
        	printf("my_cp = 0x%x, recvg_pc = 0x%x\n", my_cp, recvg_pc);
        	if (my_cp == (address_type)(-2)) my_cp = recvg_pc;

        	active_mask_t mask0, mask1;
        	for (unsigned i=0; i<tmp_active_mask.size(); i+=2) {
        		if (tmp_active_mask.test(i)) mask0.set(i);
        		if (tmp_active_mask.test(i+1)) mask1.set(i+1);
        	}
        	assert (mask0.any() || mask1.any()); // At least one thd has to be active on the TOS
        	assert (top_type == STACK_ENTRY_TYPE_NORMAL || top_type == STACK_ENTRY_TYPE_TRANS);
        	unsigned num_01 = 0;
        	if (mask0.any()) num_01 ++;
        	if (mask1.any()) num_01 ++;
        	assert (num_01 == 1 or num_01 == 2);

//			if (top_type != STACK_ENTRY_TYPE_TRANS) m_stack.pop_back();
        	m_stack.back().m_pc = my_cp;
        	m_stack.back().m_branch_div_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;

			if (mask0.any()) {
				simt_stack_entry ety0(top_epoch);
				ety0.m_pc = tmp_next_pc;
				ety0.m_active_mask = mask0;
				ety0.m_type = STACK_ENTRY_TYPE_NORMAL;
				ety0.m_recvg_pc = my_cp;
				m_stack.push_back(ety0);
			}
			if (mask1.any()) {
				simt_stack_entry ety1(top_epoch);
				ety1.m_pc = tmp_next_pc;
				ety1.m_active_mask = mask1;
				ety1.m_type = STACK_ENTRY_TYPE_NORMAL;
				ety1.m_recvg_pc = my_cp;
				m_stack.push_back(ety1);
			}

			printf("Stack after:\n");
			this->print(stdout);
			assert (m_stack.empty() || m_stack.back().m_pc != 0xFFFFFFFF);
			return;
        }

        if (next_inst_op == STAGGER_TXN_OP && top_type == STACK_ENTRY_TYPE_TRANS && getenv("IGNORE_STAGGER")==NULL &&
                		g_stagger1_count > 0) {
			g_stagger1_count --;
	        active_mask_t tmp_stagger;
			for (unsigned i=0; i<tmp_active_mask.size(); i+=2) {
				if (tmp_active_mask.test(i)) {
					tmp_stagger.set(i);
					tmp_active_mask.reset(i);
				}
			}
        	printf("---------- Stagger ----------- cycle %llu\n", gpu_sim_cycle + gpu_tot_sim_cycle);
        	assert (top_active_mask.any() == false);
        	m_stack.pop_back();

        	if (tmp_stagger.any()) {
        		simt_stack_entry ety0(top_epoch);
        		ety0.m_pc = tmp_next_pc;
        		ety0.m_active_mask = tmp_stagger;
        		ety0.m_type = STACK_ENTRY_TYPE_TRANS;
        		ety0.m_recvg_pc = top_recvg_pc;
        		ety0.m_branch_div_cycle = top_branch_div_cyc;
        		m_stack.push_back(ety0);
        	}

        	if (tmp_active_mask.any()) {
        		simt_stack_entry ety1(top_epoch);
        		ety1.m_pc = tmp_next_pc;
        		ety1.m_active_mask = tmp_active_mask;
        		ety1.m_type = STACK_ENTRY_TYPE_TRANS;
        		ety1.m_recvg_pc = top_recvg_pc;
        		ety1.m_branch_div_cycle = top_branch_div_cyc;
        		m_stack.push_back(ety1);
        	}


        	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
        	if (scc) {
        		tm_warp_info& twi = scc->m_warp[m_warp_id].get_tm_warp_info();
        		twi.backupRWSetToStack(tmp_stagger, scc->m_thread, m_warp_id, m_stack.back().m_epoch, m_stack.back().m_serial);
        	}

        	printf("Stack after stagger:\n");
			this->print(stdout);
			assert (m_stack.empty() || m_stack.back().m_pc != 0xFFFFFFFF);
        	return;
        }

        // HANDLE THE SPECIAL CASES FIRST
        if (next_inst_op == CALL_OPS)
        {
            // Since call is not a divergent instruction, all threads should have executed a call instruction
            assert(top_active_mask.any() == false);

            simt_stack_entry new_stack_entry(top_epoch);
            new_stack_entry.m_pc = tmp_next_pc;
            new_stack_entry.m_active_mask = tmp_active_mask;
            new_stack_entry.m_branch_div_cycle = gpu_sim_cycle+gpu_tot_sim_cycle;
            new_stack_entry.m_type = STACK_ENTRY_TYPE_CALL;
            m_stack.push_back(new_stack_entry);
            assert (m_stack.empty() || m_stack.back().m_pc != 0xFFFFFFFF);
            return;

        } else if (next_inst_op == RET_OPS and 
                   m_stack.size() > 1 and  // avoid popping the top-level kernel return
                   top_type == STACK_ENTRY_TYPE_CALL) 
        {
            // pop the CALL Entry
            assert(top_active_mask.any() == false);
            m_stack.pop_back();

            assert(m_stack.size() > 0);
            m_stack.back().m_pc=tmp_next_pc;// set the PC of the stack top entry to return PC from  the call stack;
            // Check if the New top of the stack is reconverging
            if (tmp_next_pc == m_stack.back().m_recvg_pc && m_stack.back().m_type!=STACK_ENTRY_TYPE_CALL)
            {
                assert(m_stack.back().m_type==STACK_ENTRY_TYPE_NORMAL);
                m_stack.pop_back();
            }
            assert (m_stack.empty() || m_stack.back().m_pc != 0xFFFFFFFF);
            return;
        }

        // discard the new entry if its PC matches with reconvergence PC
        // that automatically reconverges the entry
        // If the top stack entry is CALL, dont reconverge.
        if (tmp_next_pc == top_recvg_pc && (top_type != STACK_ENTRY_TYPE_CALL)) {/*
        	if (scc) {
        		if (m_stack.size() > 1 && m_stack[m_stack.size()-2].m_pc == top_recvg_pc
        				&& top_active_mask.none()) {
					printf("Reconverged. I need to set the correct PC for all threads.\n");
					for (unsigned t = 0; t < 32; t++) {
						ptx_thread_info* pti = scc->m_thread[m_warp_id * m_warp_size + t];
						if (pti) { pti->m_NPC = pti->m_PC = tmp_next_pc; }
					}
        		}
        	}*/
        	continue;
        }

        // this new entry is not converging
        // if this entry does not include thread from the warp, divergence occurs
        if (top_active_mask.any() && !warp_diverged ) {
            warp_diverged = true;
            // modify the existing top entry into a reconvergence entry in the pdom stack
            new_recvg_pc = recvg_pc;
            if (new_recvg_pc != top_recvg_pc) {
                m_stack.back().m_pc = new_recvg_pc;
                m_stack.back().m_branch_div_cycle = gpu_sim_cycle+gpu_tot_sim_cycle;
                m_stack.push_back(simt_stack_entry(top_epoch));
            }
        }

        // discard the new entry if its PC matches with reconvergence PC
        if (warp_diverged && tmp_next_pc == new_recvg_pc) continue;

        // update the current top of pdom stack
        m_stack.back().m_pc = tmp_next_pc;
        m_stack.back().m_active_mask = tmp_active_mask;
        if (warp_diverged) {
            m_stack.back().m_calldepth = 0;
            m_stack.back().m_recvg_pc = new_recvg_pc;
            if (thread_stagger.any()) m_stack.back().m_type = STACK_ENTRY_TYPE_TRANS;
            else m_stack.back().m_type = STACK_ENTRY_TYPE_NORMAL;
        } else {
            m_stack.back().m_recvg_pc = top_recvg_pc;
        }

        m_stack.push_back(simt_stack_entry(top_epoch));

        if (should_print && break_here) {
        	printf("one while loop finished ...\n");
        	this->print(stdout);
        }
    }

    if (getenv("TOMMY_FLAG_0808")) {
    	if (m_stack.size() >= 2) {
    		const int b = m_stack.size() - 1;
    		if (m_stack[b].m_type == STACK_ENTRY_TYPE_NORMAL &&
    			m_stack[b].m_pc != 0xFFFFFFFF &&
    			m_stack[b-1].m_type == STACK_ENTRY_TYPE_TRANS) {
    			for (unsigned t=0; t<m_warp_size; t++) {
					unsigned tid = t + m_warp_size * m_warp_id;
					ptx_thread_info* pti = scc->m_thread[tid];
					if (pti) { pti->m_PC = pti->m_NPC = m_stack[b-1].m_pc; }
				}
    			if (should_print) {
					printf("[simt_stack::update] S%uW%u Maybe coming back from reconvergence. Setting PC to %x.\n",
						scc->get_sid(), m_warp_id, m_stack[b-1].m_pc);
    			}
    		}
    	}
    }

    assert(m_stack.size() > 0);

    // 2015-09-01, when running RB Tree 10x32, 480 entries
    bool popped_entry_has_ones = (m_stack.back().m_active_mask.any());

    m_stack.pop_back();

    if (popped_entry_has_ones && m_stack.size() > 0) {
    	for (int t=0; t<m_warp_size; t++) {
			if (m_stack.back().m_active_mask.test(t)) {
				unsigned tid = t + m_warp_size * m_warp_id;
				ptx_thread_info* pti = scc->m_thread[tid];
				if (pti) { pti->m_PC = pti->m_NPC = m_stack.back().m_pc; }
			}
		}
    }

    // ``Pick-up'' stack entries with the same PC
    if (g_tommy_flag_0808_1 && scc && m_stack.size() > 1 and thread_stagger.none() and next_inst_op != COMMIT_OP) {
		tm_warp_info& twi = scc->m_warp[wid].get_tm_warp_info();
		const addr_t nextpc = m_stack.back().m_pc;
		if (twi.backup_stack.size() > 0) {
			int epoch_nolaterthan = this->m_epoch - 1;
			std::vector<int> serials;
			const int num_ety_used = twi.restoreAllRWSetsFromStackByPCAndEpoch(nextpc, scc->m_thread, m_warp_id, epoch_nolaterthan,
					&serials);
			if (num_ety_used > 0 && sid == g_print_simt_stack_sid && wid == g_print_simt_stack_wid) {
				printf("[simt_stack-%llu] [0830-change] S%uW%u before:\n",
					gpu_sim_cycle + gpu_tot_sim_cycle, sid, wid);
				this->print(stdout);
			}
			g_0830_restore_by_pc_count ++;
			int num_ety_used_1 = 0, idx_last_t = -999;

			while (true) {
				bool is_found = false;

				std::deque<simt_stack_entry>::iterator itr = m_stack.begin();
				for (int i=m_stack.size() - 2; i >= 0 && m_stack[i].m_type != STACK_ENTRY_TYPE_RETRY; i--) {
					if (m_stack[i].m_pc == nextpc && m_stack[i].m_type == STACK_ENTRY_TYPE_TRANS
						 && m_stack[i].m_epoch <= epoch_nolaterthan) {
						num_ety_used_1 ++;
						is_found = true;

						for (int j=0; j<i; j++) { itr++; }

						active_mask_t delta_mask = m_stack[i].m_active_mask;
						for (int j=0; j<delta_mask.size(); j++) {
							if (delta_mask.test(j)) { assert (m_stack.back().m_active_mask.test(j) == false); }
						}
						m_stack.back().m_active_mask |= delta_mask;

						// If the current entry is N, "propagate" the "0->1" to the nearest T entry
						if (m_stack.back().m_type == STACK_ENTRY_TYPE_NORMAL) {
							bool has_t = false;
							for (idx_last_t=m_stack.size() - 1; idx_last_t>=0; idx_last_t--)
							{
								if (m_stack[idx_last_t].m_type == STACK_ENTRY_TYPE_TRANS) { has_t = true; break; }
							}
							if (has_t) {
								if (num_ety_used > 0 && sid == g_print_simt_stack_sid && wid == g_print_simt_stack_wid) {
									printf("   propagating 1 from m_stack[%d] to m_stack[%d]\n",
										i, idx_last_t);
								}
								assert (m_stack[idx_last_t].m_type == STACK_ENTRY_TYPE_TRANS);
								for (int j=0; j<delta_mask.size(); j++) {
									if (delta_mask.test(j))
										m_stack[idx_last_t].m_active_mask.set(j);
								}
							}
						}

						m_stack.erase(itr);
						break;
					}
				}

				if (!is_found) break;
			}
			if (num_ety_used_1 != num_ety_used) {
				printf("Oh! Error! Num of Etys Used: %d vs %d, pc=%x, epoch_nolaterthan=%d\n",
						num_ety_used_1, num_ety_used, nextpc, epoch_nolaterthan);
				printf("SN's used:");
				for (int x=0; x<serials.size(); x++) {
					printf(" %d", serials[x]);
				}
				printf("\n");
				printf("SN's in the current stack:");
				for (int i=0; i<m_stack.size(); i++) {
					printf(" %d", m_stack[i].m_serial);
				}
				printf("Epochs in the current stack:\n");
				for (int i=0; i<m_stack.size(); i++) {
					printf(" %d", m_stack[i].m_epoch);
				}
				printf("\n");
				print(stdout);
			}
			assert (num_ety_used_1 == num_ety_used);
			if (num_ety_used > 0 && sid == g_print_simt_stack_sid && wid == g_print_simt_stack_wid) {
				printf("[simt_stack-%llu] [0830-change] S%uW%u %d etys used\n",
						gpu_sim_cycle + gpu_tot_sim_cycle, sid, wid, num_ety_used);
				printf("[simt_stack-%llu] [0830-change] S%uW%u after:\n",
						gpu_sim_cycle + gpu_tot_sim_cycle, sid, wid);
				printf("[simt_stack-%llu] [0830-change] idx_last_t=%d\n", idx_last_t);
				this->print(stdout);
			}
		}

		// Also we need to set PC
		for (int t=0; t<m_warp_size; t++) {
			if (m_stack.back().m_active_mask.test(t)) {
				unsigned tid = t + m_warp_size * m_warp_id;
				ptx_thread_info* pti = scc->m_thread[tid];
				if (pti) { pti->m_PC = pti->m_NPC = m_stack.back().m_pc; }
			}
		}
    }

    if (break_here && should_print) {
    	printf("After the whole while loop finished ...\n");
    	this->print(stdout);
    	printf("PCs of this warp:\n");
    	for (unsigned t=0; t<m_warp_size; t++) {
    		unsigned tid = t + m_warp_size * m_warp_id;
    		ptx_thread_info* pti = scc->m_thread[tid];
    		if (pti) { printf(" %u:%x", t, pti->m_PC); }
    	}
    	printf("\n");
    }

    if (warp_diverged) {
        ptx_file_line_stats_add_warp_divergence(top_pc, 1); 
    }

    if (thread_stagger.any() && should_print) {
    	printf("Stack-after w/ stagger:\n");
    	this->print(stdout);

    	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
		if (scc) {
			int wid = m_warp_id;
			tm_warp_info& twi = scc->m_warp[wid].get_tm_warp_info();
			printf("Read log size: %u, write log size: %u:\n", twi.m_read_log_size, twi.m_write_log_size);
		}
    }
    assert (m_stack.empty() || m_stack.back().m_pc != 0xFFFFFFFF);

/*
	for (unsigned t = 0; t < 32; t++) {
		if (scc) {
			address_type pc3 = m_stack.back().m_pc;
			ptx_thread_info* pti = scc->m_thread[m_warp_id * m_warp_size + t];
			if (pti) { pti->m_NPC = pti->m_PC = pc3; }
		}
	}
	*/
}

extern std::set<new_addr_type> stagger_block_addrs;
bool Cartographer_lookupPTXSource(unsigned pc, std::string* sz);
void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpId)
{
	functionalCoreSim* fcs = dynamic_cast<functionalCoreSim*> (this);

	int sid = -999;
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(this);
	bool should_print = false;
	if (scc) sid = scc->get_sid();
	if (sid == g_print_simt_stack_sid && inst.warp_id() == g_print_simt_stack_wid) should_print = true;

	bool break_flag = false;
	if (g_tommy_break_cycle <= gpu_sim_cycle + gpu_tot_sim_cycle) { break_flag = true; }

	char* x = getenv("TOMMYDBG");
	if (break_flag && should_print) { // Why not triggered?
		char tmp[333];
		std::string ptxsource;
		if (Cartographer_lookupPTXSource(inst.pc, &ptxsource)) {
			sprintf(tmp, "%s", ptxsource.c_str());
		} else {
			sprintf(tmp, "No PTX Info");
		}
		printf("[TommyDbg] S%uW%u PC=%x [", sid, inst.warp_id(), inst.pc);
		const active_mask_t& am = inst.get_active_mask();
		for (unsigned i=0; i<am.size(); i++) {
			if (am.test(i)) printf("1"); else printf("0");
		}
		printf("] warp instruction: %s", tmp);
		printf(", is_tx:%d\n", inst.in_transaction);
	}

	stagger_block_addrs.clear();
    for ( unsigned t=0; t < m_warp_size; t++ ) {
        if( inst.active(t) ) {
            if(warpId==(unsigned (-1)))
                warpId = inst.warp_id();
            unsigned tid=m_warp_size*warpId+t;
            m_thread[tid]->ptx_exec_inst(inst,t);
            //virtual function
            checkExecutionStatusAndUpdate(inst,t,tid);
        }
    }
}
  
// void simt_stack::clone_entry(unsigned dst, unsigned src) 
// {
//     m_pc[dst] = m_pc[src]; 
//     m_active_mask[dst] = m_active_mask[src]; 
//     m_recvg_pc[dst] = m_recvg_pc[src]; 
//     m_calldepth[dst] = m_calldepth[src]; 
//     m_branch_div_cycle[dst] = m_branch_div_cycle[src]; 
//     m_type[dst] = m_type[src]; 
// }

tm_parallel_pdom_warp_ctx_t::tm_parallel_pdom_warp_ctx_t( unsigned wid, unsigned warp_size, core_t* core )
    : simt_stack(wid, warp_size, core)
{ }

void tm_parallel_pdom_warp_ctx_t::txbegin(address_type tm_restart_pc) 
{
    if (m_in_transaction) return; 

    m_epoch ++;

    assert(m_stack.size() < m_warp_size * 2 - 2); // not necessarily true with infinite recursion, but a good check anyway

    const simt_stack_entry &orig_top_entry = m_stack.back(); 

    m_active_mask_at_txbegin = orig_top_entry.m_active_mask; 

    // insert retry entry
    simt_stack_entry retry_entry(orig_top_entry); 
    // unsigned retry_idx = m_stack_top + 1;
    // clone_entry(retry_idx, m_stack_top); 
    retry_entry.m_pc = tm_restart_pc; 
    retry_entry.m_recvg_pc = -1; //HACK: need to set this to the insn after txcommit() 
    retry_entry.m_active_mask.reset(); 
    retry_entry.m_type = STACK_ENTRY_TYPE_RETRY;
    retry_entry.m_epoch = this->m_epoch;
    m_stack.push_back(retry_entry); 

    // insert transaction entry 
    simt_stack_entry texec_entry(orig_top_entry); 
    // unsigned texec_idx = m_stack_top + 2;
    // clone_entry(texec_idx, m_stack_top); 
    texec_entry.m_recvg_pc = -1; 
    texec_entry.m_pc = tm_restart_pc; 
    texec_entry.m_recvg_pc = -1; //HACK: need to set this to the insn after txcommit() 
    texec_entry.m_type = STACK_ENTRY_TYPE_TRANS; 
    texec_entry.m_epoch = this->m_epoch;
    m_stack.push_back(texec_entry); 

//    printf("[%llu] Parallel SIMT Stack txbegin. Original Mask: ");
//    PrintSIMTStackEntry(&(orig_top_entry.m_active_mask));
//    printf("\n Retry Mask: ");
//    PrintSIMTStackEntry(&(retry_entry.m_active_mask));
//    printf(", TXExec Mask: ");
//    PrintSIMTStackEntry(&(texec_entry.m_active_mask));
//    printf("\n");

    CartographerTimeSeries_onWarpActiveMaskChanged(m_core, m_warp_id, &(texec_entry.m_active_mask), "TXBegin");

    //TODO: set TOS entry's PC to the insn after txcommit() 

    m_in_transaction = true;
}

void tm_parallel_pdom_warp_ctx_t::txrestart() 
{
	m_epoch ++;

	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
    assert(m_in_transaction); 

    // unsigned retry_idx = m_stack_top; 
    simt_stack_entry &retry_entry = m_stack.back(); 
    assert(retry_entry.m_type == STACK_ENTRY_TYPE_RETRY); 
    assert(retry_entry.m_active_mask.any()); 

    // clone the retry entry to create a new top level transaction entry 
    // unsigned texec_idx = retry_idx + 1; 
    // clone_entry(texec_idx, retry_idx); 
    simt_stack_entry texec_entry(retry_entry); 
    texec_entry.m_type = STACK_ENTRY_TYPE_TRANS;
    texec_entry.m_epoch = this->m_epoch;
    retry_entry.m_epoch = this->m_epoch;

    // reset active mask in retry entry 
    retry_entry.m_active_mask.reset();

    if (scc) {
		for (unsigned t = 0; t < m_warp_size; t++) {
			if (texec_entry.m_active_mask.test(t)) {
				unsigned tid = t + m_warp_id * m_warp_size;
				ptx_thread_info* pti = scc->m_thread[tid];
				pti->m_PC = pti->m_NPC = texec_entry.m_pc;
			}
		}
    }

    CartographerTimeSeries_onWarpActiveMaskChanged(m_core, m_warp_id, &(texec_entry.m_active_mask), "TXRestart");

    m_stack.push_back(texec_entry); 
}

// check for correctness conditions when a warp-level transaction is restarted 
bool tm_parallel_pdom_warp_ctx_t::check_txrestart_warp_level() 
{
   bool valid = true; 

   // all threads in warp restarts 
   assert(m_stack.back().m_active_mask == m_active_mask_at_txbegin); 
   // all threads in warp restarts 
   assert(m_stack.back().m_type == STACK_ENTRY_TYPE_TRANS); 
   return valid; 
}

// check for correctness conditions when a warp-level transaction is committed 
bool tm_parallel_pdom_warp_ctx_t::check_txcommit_warp_level() 
{
   bool valid = true; 

   // warp should no longer in transaction 
   assert(m_stack.back().m_type == STACK_ENTRY_TYPE_NORMAL); 
   return valid; 
}

void tm_parallel_pdom_warp_ctx_t::txabort(unsigned thread_id) 
{
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
	bool flag_0808 = (getenv("TOMMY_FLAG_0808") != NULL);
	flag_0808 |= (getenv("STAGGER1_COUNT") != NULL);
    assert(m_in_transaction); 
    unsigned wtid = thread_id % m_warp_size, wid = thread_id / m_warp_size;


	bool should_print = false;
	int sid = -1;
	if (scc) {
		if (scc->get_sid()==g_print_simt_stack_sid && m_warp_id==g_print_simt_stack_wid) {
			should_print=true; sid = scc->get_sid();
		}
	}

	if (should_print) {
		printf("[%llu][tm_parallel_pdom_warp_ctx_t::txabort] S%uW%u SIMT Stack at txabort of tid=%u :\n",
			gpu_sim_cycle + gpu_tot_sim_cycle, sid, m_warp_id, thread_id);
		this->print(stdout);
	}

    // mask out this thread in the active mask of all entries in the transaction
    // pop TOS entry if the active mask is empty 
    int idx; 

    if (g_tommy_flag_0808_1) {
		for (idx = m_stack.size() - 1; idx > 0 and m_stack[idx].m_type == STACK_ENTRY_TYPE_TRANS; idx --) {
			active_mask_t tmp_mask = m_stack[idx].m_active_mask;
			tmp_mask.reset(wtid);
			if (tmp_mask.none()) {
				reorderTEntry(thread_id % m_warp_size);
				break;
			}
		}
    }

    assert(m_stack.size() > 0);
    unsigned count = 0;
    bool has_ever_popped = false;
    for (idx = m_stack.size() - 1; idx > 0 and m_stack[idx].m_type != STACK_ENTRY_TYPE_RETRY; idx--) {
		count ++;
        m_stack[idx].m_active_mask.reset(wtid);
        address_type pc2;
        ptx_thread_info* pti = scc->m_thread[thread_id];
		if (m_stack[idx].m_type == STACK_ENTRY_TYPE_TRANS) {
		   simt_stack_entry* rety = &(m_stack[idx-1]);
		   for (int idx1 = idx-1; idx1 >= 0; idx1--) {
			   if (m_stack[idx1].m_type == STACK_ENTRY_TYPE_RETRY) rety = &(m_stack[idx1]);
		   }
		   pti->m_PC = pti->m_NPC = pc2 = rety->m_pc;
		} else if (m_stack[idx].m_type == STACK_ENTRY_TYPE_NORMAL) {
		   pti->m_PC = pti->m_NPC = pc2 = m_stack[idx-1].m_pc;
		}

        if (m_stack[idx].m_active_mask.none()) {

           bool format_brush_flag = false;
           if (m_stack[idx].m_type == STACK_ENTRY_TYPE_NORMAL && m_stack[idx-1].m_type == STACK_ENTRY_TYPE_NORMAL) format_brush_flag = true;
           if (m_stack[idx].m_type == STACK_ENTRY_TYPE_NORMAL && m_stack[idx-1].m_type == STACK_ENTRY_TYPE_TRANS) format_brush_flag = true;
           if (m_stack[idx].m_type == STACK_ENTRY_TYPE_TRANS && m_stack[idx-1].m_type == STACK_ENTRY_TYPE_RETRY) {
        	   if (m_stack[idx-1].m_active_mask.count() > 0) // Must be retrying for this to be set true
        	   format_brush_flag = true;
           }
           m_stack[idx].m_type = STACK_ENTRY_TYPE_INVALID; // this could be an inactive entry waiting for execution
           if (idx == (int)(m_stack.size() - 1)) {
        	   if (format_brush_flag) {
				   for (unsigned t=0; t<m_warp_size; t++) {
					   unsigned tid = t + m_warp_size * wid;
					   ptx_thread_info* pti = scc->m_thread[tid];
					   if (pti) pti->m_PC = pti->m_NPC = pc2;
				   }
        	   }
        	   m_stack.pop_back();
        	   has_ever_popped = true;
           }
        }
    }

    // set the active mask for this thread at the retry entry 
    assert(m_stack[idx].m_type == STACK_ENTRY_TYPE_RETRY);
    m_stack[idx].m_active_mask.set(wtid); 

    // if the whole warp is aborted, trigger a retry 
    if (m_stack.back().m_type == STACK_ENTRY_TYPE_RETRY) {
        txrestart(); 
    }

    //TOMMY GUN
    if (getenv("TOMMY_FLAG_0808") && scc && m_stack.back().m_type == STACK_ENTRY_TYPE_TRANS && has_ever_popped) {
		for (unsigned t=0; t<m_warp_size; t++) {
			unsigned tid = t + m_warp_id * m_warp_size;
			ptx_thread_info* pti = scc->m_thread[tid];
			if (pti) { pti->m_PC = pti->m_NPC = m_stack.back().m_pc; }
		}
    }
}

// The insertion point should be the T entry that is to be removed. Example:
//
//  0  10000000 R
//  1  01100000 T
//  2  00010000 T
//  3  00001000 T <-- Will be removed
//
//  And, 1 and 2 have to be swapped. The insertion point is '3', not the guy after 3 !!!
//
//  the last lane is the last active lane in the topmost T entry
//
void simt_stack::reorderTEntry(int the_last_lane) {
	 // We should swap the TRANS entries: the bottom-of-stack should be moved to the first
	int idx_first_t = -999, num_t = 0;
	std::deque<simt_stack_entry>::iterator itr_first_t = m_stack.begin();
	for (int i=0; i<m_stack.size(); i++) {
		if (m_stack[i].m_type == STACK_ENTRY_TYPE_TRANS) {
			if (idx_first_t == -999) {
				idx_first_t = i;
			}
			num_t ++;
		}
		if (idx_first_t == -999) itr_first_t ++;
	}
	assert (itr_first_t->m_type == STACK_ENTRY_TYPE_TRANS);
	if (num_t >= 3) { // If there are 3+ T entries, the bottom one must not be all zeroes.

//		printf("simt_stack::reorderTEntry before:\n");
//		this->print(stdout);

		assert (itr_first_t->m_active_mask.count() > 0);
		simt_stack_entry t_entry = m_stack[idx_first_t];
		m_stack.erase(itr_first_t);

		// Find the place to put the "bottom-of-stack" entry
		// The place may be: 1) the first all-inactive T entry (for use with txabort)
		//                   2) Top of stack
		std::deque<simt_stack_entry>::iterator insert_point, tmp = m_stack.begin();
		active_mask_t insert_point_mask;
		for (; tmp != m_stack.end(); tmp ++) {
			if (tmp->m_type == STACK_ENTRY_TYPE_TRANS) {
				insert_point = tmp; // We can grab the last T entry
				insert_point_mask = tmp->m_active_mask;
			}
		}
		if (the_last_lane != -999) {
			if (the_last_lane < 0 || the_last_lane >= insert_point_mask.size()) {
				printf("This 'the_last_lane' seems to be invalid (it is %d).\n", the_last_lane);
			}
			bool is_ok = false;
			if (
				(insert_point_mask.count() == 1 && insert_point_mask.test(the_last_lane) == true) ||
				(insert_point_mask.count() == 0) ) { is_ok = true; }
			if (!is_ok) {
				printf("the_last_lane is %d\n", the_last_lane);
				this->print(stdout);
				assert(0);
			}
		}
		m_stack.insert(insert_point, t_entry);

		// Clear INVALID entries
		while (true) {
			bool is_found = false;
			for (std::deque<simt_stack_entry>::iterator itr = m_stack.begin();
				itr != m_stack.end(); itr++) {
				if (itr->m_type == STACK_ENTRY_TYPE_INVALID) {
					is_found = true;
					m_stack.erase(itr);
					break;
				}
			}
			if (!is_found) break;
		}

//		printf("simt_stack::reorderTEntry after:\n");
//				this->print(stdout);
	}
}

void tm_parallel_pdom_warp_ctx_t::txcommit(unsigned thread_id, address_type tm_commit_pc)
{
	bool should_print = false;
	int sid = -1;
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
	if (scc) {
		sid = scc->get_sid();
		if (scc->get_sid()==g_print_simt_stack_sid && m_warp_id==g_print_simt_stack_wid) {
			should_print=true;
		}
	}

	if (should_print) {
		printf("[%llu][tm_parallel_pdom_warp_ctx_t::txcommit] S%uW%u SIMT Stack at txcommit of tid=%u :\n",
			gpu_sim_cycle + gpu_tot_sim_cycle, sid, m_warp_id, thread_id);
		this->print(stdout);
	}

    assert(m_in_transaction);

    unsigned int stack_top = m_stack.size() - 1;
    bool ok = true;

    bool allow_multiT = false;
    if (getenv("TOMMY_FLAG_0808") || getenv("STAGGER1_COUNT")) allow_multiT = true;

    if (allow_multiT == false) {
		if (! ((m_stack[stack_top - 1].m_type == STACK_ENTRY_TYPE_RETRY) &&
			   (m_stack.back().m_type == STACK_ENTRY_TYPE_TRANS)) ) ok = false;
    } else {
		for (int i = stack_top; i >= 0; i--) {
			if (m_stack[i].m_type == STACK_ENTRY_TYPE_TRANS) {
				if (i < stack_top) {
					if (m_stack[i+1].m_type != STACK_ENTRY_TYPE_TRANS) {
						ok = false; break;
					}
				}
			}
			else if (m_stack[i].m_type == STACK_ENTRY_TYPE_RETRY) {
				if (i == stack_top) { ok = false; break; }
				else {
					if (m_stack[i+1].m_type != STACK_ENTRY_TYPE_TRANS) {
						ok = false; break;
					}
				}
			}
		}
    }

    if ( ok == false ) {
    	printf("Ouch! txcommit's assertion on STACK ENTRY TYPE failed. S%uW%u, cycle=%llu\n", sid, m_warp_id,
    			gpu_sim_cycle + gpu_tot_sim_cycle);
    	printf("-----------------------------8<---------------------------\n");
    	this->print(stdout);
    	printf("----------------------------->8---------------------------\n");
    	g_tm_global_statistics.print(stdout);
    	assert(0);
    }

    unsigned wtid = thread_id % m_warp_size; 

    // clear the active mask for this thread at the top level transaction entry 
    m_stack.back().m_active_mask.reset(wtid); 

    int idx_retry = m_stack.size()-1;
    for (; idx_retry >= 0 && m_stack[idx_retry].m_type != STACK_ENTRY_TYPE_RETRY; idx_retry --);
    assert (idx_retry >= 0);

    if (m_stack.back().m_active_mask.none()) {

        if (g_tommy_flag_0808_1) { reorderTEntry(thread_id); }
        assert (m_stack.back().m_active_mask.none()); // Make sure reorder doesn't reorder away the noneness

        // all threads in this warp commited or aborted 
        // pop this transaction entry 
        m_stack.pop_back(); 

        // Has to be executed when a stack entry is popped

        /*
        if (getenv("TOMMY_FLAG_0808")) {
			while (m_stack.size() > 0 && m_stack.back().m_type == STACK_ENTRY_TYPE_TRANS) {
				simt_stack_entry& back_entry = m_stack.back();
				active_mask_t&    back_mask  = back_entry.m_active_mask;

				for (unsigned i=0; i<back_mask.size(); i++) {
					if (back_mask.test(i)) {
						unsigned tid = thread_id - (thread_id % m_warp_size) + i;
						value_based_tm_manager* tmm = dynamic_cast<value_based_tm_manager*>(m_core->m_thread[tid]->get_tm_manager());

						if (tmm) {
							bool old_m_violated = tmm->m_violated;
							tmm->validate();
							bool new_m_violated = tmm->m_violated;
							tmm->m_violated = old_m_violated;

							if (new_m_violated) {
								printf("Stagger-Aborted tid=%u\n", tid);
								tmm->abort();
								back_mask.reset(i);
								assert (m_stack[idx_retry].m_active_mask.test(i) == false);
								m_stack[idx_retry].m_active_mask.set(i);
								g_num_stagger_aborts ++;
							}
						}
					}
				}

				if (back_mask.any()) break;
				else m_stack.pop_back();
			}
        }*/

        // check for need to restart 
        const simt_stack_entry &retry_entry = m_stack.back();
        if (retry_entry.m_type == STACK_ENTRY_TYPE_RETRY) {
			if (retry_entry.m_active_mask.any()) {
				txrestart();
			} else {
				// no restart needed, pop the retry entry and set pc of TOS to the
				// next insn after commit, transaction is done for this warp
				m_stack.pop_back();
				m_stack.back().m_pc = tm_commit_pc;

				if (/*getenv("TOMMY_FLAG_0808") && */scc) { // Applies to FLAG0808==false also.
					for (unsigned t=0; t<m_warp_size; t++) {
						unsigned tid = t + m_warp_id * m_warp_size;
						ptx_thread_info* pti = scc->m_thread[tid];
						if (pti) {
							address_type first_active_pc = 0xFFFFFFFF;
							for (int x = m_stack.size()-1; x>=0; x--) {
								if (m_stack[x].m_active_mask.test(t)) {
									first_active_pc = m_stack[x].m_pc; break;
								}
							}
							assert (first_active_pc != 0xFFFFFFFF);
							pti->m_PC = pti->m_NPC = first_active_pc;
						}
					}
				}

				// if the next pc after commit happens to be the reconvergence point as well
				// this is no longer handled by normal stack handler because functional commit is now further down the pipeline
				while ( m_stack.back().at_recvg() ) {
				   m_stack.pop_back();
				}
				CartographerTimeSeries_onWarpActiveMaskChanged(m_core, m_warp_id, &(m_stack.back().m_active_mask), "TxCommit");
				m_in_transaction = false;

				/*
				shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(m_core);
				if (scc && m_stack.size() > 0) {
					for (unsigned t=0; t<m_warp_size; t++) {
						unsigned tid = t + m_warp_id * m_warp_size;
						ptx_thread_info* pti = scc->m_thread[tid];
						if (pti) pti->m_PC = pti->m_NPC = m_stack.back().m_pc;
					}
				}
				*/
			}
        } else {
        	// Get prepared to restore R/W set
        }
    }

    if (getenv("TOMMY_FLAG_0808") && scc && m_stack.back().m_type == STACK_ENTRY_TYPE_TRANS) {
		for (unsigned t=0; t<m_warp_size; t++) {
			unsigned tid = t + m_warp_id * m_warp_size;
			ptx_thread_info* pti = scc->m_thread[tid];
			if (pti) { pti->m_PC = pti->m_NPC = m_stack.back().m_pc; }
		}
    }

    if (should_print) {
		printf("After txcommit:\n");
		this->print(stdout);
		printf("PCs of this warp: ");
		for (unsigned t = 0; t < m_warp_size; t++) {
			unsigned tid = t + m_warp_size * m_warp_id;
			ptx_thread_info* pti = scc->m_thread[tid];
			if (pti) printf(" %u:%x", t, pti->m_PC);
		}
		printf("\n");
    }
}

tm_serial_pdom_warp_ctx_t::tm_serial_pdom_warp_ctx_t( unsigned wid, unsigned warp_size, core_t* core )
    : simt_stack(wid, warp_size, core)
{ }

void tm_serial_pdom_warp_ctx_t::txbegin(address_type tm_restart_pc) 
{
    if (m_in_transaction) return; 
    assert(m_stack.size() < m_warp_size * 2 - 2); // not necessarily true with infinite recursion, but a good check anyway

    const simt_stack_entry &orig_top_entry = m_stack.back(); 

    // insert retry entry 
    // - this holds the threads that are deferred due to serialization, 
    // as well as place holder for transaction retry 
    // (though usually the same thread will set to retry after abortion)
    simt_stack_entry retry_entry(orig_top_entry); 
    retry_entry.m_pc = tm_restart_pc; 
    retry_entry.m_recvg_pc = -1; //HACK: need to set this to the insn after txcommit() 
    retry_entry.m_type = STACK_ENTRY_TYPE_RETRY; 
    m_stack.push_back(retry_entry); 
    unsigned retry_idx = m_stack.size() - 1; 

    // insert transaction entry 
    simt_stack_entry texec_entry(orig_top_entry); 
    texec_entry.m_recvg_pc = -1; 
    texec_entry.m_pc = tm_restart_pc; 
    texec_entry.m_recvg_pc = -1; //HACK: need to set this to the insn after txcommit() 
    texec_entry.m_type = STACK_ENTRY_TYPE_TRANS; 
    m_stack.push_back(texec_entry); 
    unsigned texec_idx = m_stack.size() - 1; 



    //TODO: set TOS entry's PC to the insn after txcommit() 

    tx_start_thread(retry_idx, texec_idx); 

    m_in_transaction = true; 
}

// move one thread from retry to transaction entry 
void tm_serial_pdom_warp_ctx_t::tx_start_thread(unsigned retry_idx, unsigned texec_idx) 
{
    // find a thread in the retry entry and bring it to transaction entry 
    m_stack[texec_idx].m_active_mask.reset(); // clear mask in transaction entry 
    for (unsigned t = 0; t < m_warp_size; t++) {
        if (m_stack[retry_idx].m_active_mask.test(t) == true) {
            m_stack[texec_idx].m_active_mask.set(t); 
            m_stack[retry_idx].m_active_mask.reset(t);
            
            // unsigned hw_thread_id = t + m_warp_id * m_warp_size; 
            // m_shader->set_transactional_thread( m_shader->get_func_thread_info(hw_thread_id) );

            break; 
        }
    }
    assert(m_stack[texec_idx].m_active_mask.any()); 
}

void tm_serial_pdom_warp_ctx_t::txrestart() 
{
    assert(m_in_transaction); 

    // unsigned retry_idx = m_stack_top; 
    simt_stack_entry &retry_entry = m_stack.back(); 
    assert(retry_entry.m_type == STACK_ENTRY_TYPE_RETRY); 
    assert(retry_entry.m_active_mask.any()); 
    unsigned retry_idx = m_stack.size() - 1;

    // clone the retry entry to create a new top level transaction entry 
    // unsigned texec_idx = retry_idx + 1; 
    // clone_entry(texec_idx, retry_idx); 
    simt_stack_entry texec_entry(retry_entry); 
    texec_entry.m_type = STACK_ENTRY_TYPE_TRANS; 
    m_stack.push_back(texec_entry); 
    unsigned texec_idx = m_stack.size() - 1;

    tx_start_thread(retry_idx, texec_idx); 
    // no need to reset active mask in retry entry 
}

void tm_serial_pdom_warp_ctx_t::txabort(unsigned thread_id) 
{
    assert(m_in_transaction); 
    unsigned wtid = thread_id % m_warp_size; 

    // mask out this thread in the active mask of all entries in the transaction
    // pop TOS entry if the active mask is empty 
    int idx; 
    assert(m_stack.size() > 0);
    for (idx = m_stack.size() - 1; idx > 0 and m_stack[idx].m_type != STACK_ENTRY_TYPE_RETRY; idx--) {
        m_stack[idx].m_active_mask.reset(wtid); 
        if (m_stack[idx].m_active_mask.none() and idx == (int)(m_stack.size() - 1)) {
            m_stack.pop_back(); 
        }
    }

    // set the active mask for this thread at the retry entry 
    assert(m_stack[idx].m_type == STACK_ENTRY_TYPE_RETRY);
    m_stack[idx].m_active_mask.set(wtid); 

    // if the whole warp is aborted, trigger a retry 
    if (m_stack.back().m_type == STACK_ENTRY_TYPE_RETRY) {
        txrestart(); 
    }
}

void tm_serial_pdom_warp_ctx_t::txcommit(unsigned thread_id, address_type tm_commit_pc)
{
    assert(m_in_transaction); 
    unsigned int stack_top = m_stack.size() - 1; 
    assert(m_stack[stack_top - 1].m_type == STACK_ENTRY_TYPE_RETRY); 
    assert(m_stack.back().m_type == STACK_ENTRY_TYPE_TRANS); 

    unsigned wtid = thread_id % m_warp_size; 

    // clear the active mask for this thread at the top level transaction entry 
    m_stack.back().m_active_mask.reset(wtid); 

    // m_shader->set_transactional_thread(NULL);

    if (m_stack.back().m_active_mask.none()) {
        // all threads in this warp commited or aborted 
        // pop this transaction entry 
        m_stack.pop_back(); 

        // check for need to restart 
        const simt_stack_entry &retry_entry = m_stack.back(); 
        if (retry_entry.m_active_mask.any()) {
            txrestart(); 
        } else {
            // no restart needed, pop the retry entry and set pc of TOS to the 
            // next insn after commit, transaction is done for this warp 
            m_stack.pop_back(); 
            m_stack.back().m_pc = tm_commit_pc; 
            m_in_transaction = false; 
        }
    }
}

bool  core_t::ptx_thread_done( unsigned hw_thread_id ) const  
{
    return ((m_thread[ hw_thread_id ]==NULL) || m_thread[ hw_thread_id ]->is_done());
}

void core_t::updateSIMTStack(unsigned warpId, warp_inst_t * inst)
{
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(this);
	int sid = -1, wid = inst->warp_id();
	if (scc) { sid = scc->m_sid; }
	bool should_print = false;
	if (g_print_simt_stack_sid == sid && g_print_simt_stack_wid == wid)
		should_print = true;

    // extract thread done and next pc information from functional model here
    simt_mask_t thread_done;
    addr_vector_t next_pc;
    unsigned wtid = warpId * m_warp_size;
    for (unsigned i = 0; i < m_warp_size; i++) {
        if( ptx_thread_done(wtid+i) ) {
            thread_done.set(i);
            next_pc.push_back( (address_type)-1 );
        } else {
            assert( m_thread[wtid + i] != NULL ); 
            if( inst->reconvergence_pc == RECONVERGE_RETURN_PC ) 
                inst->reconvergence_pc = get_return_pc(m_thread[wtid+i]);
            next_pc.push_back( m_thread[wtid+i]->get_pc() );
        }
    }
    std::bitset<MAX_WARP_SIZE_SIMT_STACK> before, after;
    bool has_before = false, has_after = false;
    if (m_simt_stack[warpId]->m_stack.size() > 0) {
    	before = m_simt_stack[warpId]->get_active_mask();
    	has_before = true;
    }

	if (should_print && gpu_sim_cycle + gpu_tot_sim_cycle > g_tommy_break_cycle) {
		printf("[updateSIMTStack-%llu] on entry:\n", gpu_sim_cycle + gpu_tot_sim_cycle);
		m_simt_stack[warpId]->print(stdout);
		printf("next_pc:");
		for (addr_vector_t::iterator itr = next_pc.begin(); itr != next_pc.end(); itr++) {
			printf(" %x", *itr);
		}
		printf("\n");
	}

    m_simt_stack[warpId]->update(thread_done, inst->get_stagger_mask(), next_pc, inst->reconvergence_pc, inst->op, inst->pc);
    if (g_tommy_dbg_0830) { m_simt_stack[warpId]->sanityCheck0830(); }

	// stack sanity chk
	for (unsigned i=0; i<m_simt_stack[warpId]->m_stack.size(); i++) {
		if (m_simt_stack[warpId]->m_stack.at(i).m_pc == 0xFFFFFFFF) {
			printf("S%uW%u\n", sid, warpId);
			printf("See an pc of 0xFFFFFFFF at cycle %llu\n", gpu_sim_cycle + gpu_tot_sim_cycle);
			m_simt_stack[warpId]->print(stdout);
			printf("Next PC:");
			for (addr_vector_t::iterator itr = next_pc.begin(); itr != next_pc.end(); itr++) {
				printf(" %x", *itr);
			}
			printf("\n");
			printf("Stagger Mask: ");
			for (unsigned i=0; i<inst->get_stagger_mask().size(); i++) {
				if (inst->get_stagger_mask().test(i)) printf("1"); else printf("0");
			}
			printf("\n");
			printf("Thread Done: ");
			for (unsigned i=0; i<thread_done.size(); i++) {
				if (thread_done.test(i)) printf("1"); else printf("0");
			}
			printf("\n");
			assert(0);
		}
	}

    if (m_simt_stack[warpId]->m_stack.size() > 0) {
    	after  = m_simt_stack[warpId]->get_active_mask();
    	has_after  = true;
    }

    if (has_after) {
    	CartographerTimeSeries_onWarpActiveMaskChanged(this, warpId, &after, "updateSIMTStack");
    }

    char tmp[333];
    std::string ptxsource;
    if (Cartographer_lookupPTXSource(inst->pc, &ptxsource)) {
    	sprintf(tmp, "%s", ptxsource.c_str());
    } else {
    	sprintf(tmp, "No PTX Info");
    }

    if (should_print && g_tommy_break_cycle <= gpu_sim_cycle + gpu_tot_sim_cycle) {
    	char in_commit = '?';
    	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(this);
    	int sid = -1;

    	if (scc) {
    		in_commit = scc->m_scoreboard->inTxCommit(warpId) ? 'Y' : 'N';
    		sid = scc->get_sid();
    	}

		printf("\n[updateSIMTStack-%llu] inst.pc=%x, \'%s\', SIMT Stack of (SID %d, Warp %u) [InCommit: %c] [LogSize R=%d W=%d]\n",
				gpu_sim_cycle, inst->pc,
				tmp, sid,
				warpId, in_commit,
				scc->m_warp[wid].get_tm_warp_info().m_read_log_size,
				scc->m_warp[wid].get_tm_warp_info().m_write_log_size);
		if (has_before) {
			printf("[");
			for (unsigned i=0; i<before.size(); i++) {
				if (before.test(i)) { printf("1"); }
				else                { printf("0"); }
			}
		} else { printf("[ No before-stack "); }
		printf("] ---> [");
		if (has_after) {
			for (unsigned i=0; i<after.size(); i++) {
				if (after.test(i)) { printf("1"); }
				else               { printf("0"); }
			}
		} else { printf(" No after-stack "); }
		printf("]\n");


		m_simt_stack[warpId]->print(stdout);
    }
}

//! Get the warp to be executed using the data taken form the SIMT stack
warp_inst_t core_t::getExecuteWarp(unsigned warpId)
{
    unsigned pc,rpc;
    m_simt_stack[warpId]->get_pdom_stack_top_info(&pc,&rpc);
    warp_inst_t wi= *ptx_fetch_inst(pc);
    wi.set_active(m_simt_stack[warpId]->get_active_mask());
    return wi;
}

void core_t::deleteSIMTStack()
{
    if ( m_simt_stack ) {
        for (unsigned i = 0; i < m_warp_count; ++i) 
            delete m_simt_stack[i];
        delete[] m_simt_stack;
        m_simt_stack = NULL;
    }
}

void core_t::initilizeSIMTStack(unsigned warp_count, unsigned warp_size)
{ 
    m_simt_stack = new simt_stack*[warp_count];
    for (unsigned i = 0; i < warp_count; ++i) {
        // m_simt_stack[i] = new simt_stack(i,warp_size);
        if (m_gpu->getShaderCoreConfig()->tm_serial_pdom_stack == true) {
            m_simt_stack[i] = new tm_serial_pdom_warp_ctx_t(i, warp_size, this);
        } else {
            m_simt_stack[i] = new tm_parallel_pdom_warp_ctx_t(i, warp_size, this);
        }
    }
}

void core_t::get_pdom_stack_top_info( unsigned warpId, unsigned *pc, unsigned *rpc ) const
{
    m_simt_stack[warpId]->get_pdom_stack_top_info(pc,rpc);
}

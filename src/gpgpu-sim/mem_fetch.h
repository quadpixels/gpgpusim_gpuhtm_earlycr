// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include "addrdec.h"
#include "../abstract_hardware_model.h"
#include <bitset>

enum mf_type {
    READ_REQUEST = 0,
    WRITE_REQUEST,
    READ_REPLY, // send to shader
    WRITE_ACK,

    // value-based TM
    TX_CU_ALLOC,
    TX_READ_SET,
    TX_WRITE_SET,
    TX_DONE_FILL,
    TX_SKIP,
    TX_PASS,
    TX_FAIL,
    CU_PASS,
    CU_FAIL,
    CU_ALLOC_PASS,
    CU_ALLOC_FAIL,
    CU_DONE_COMMIT,

    TR_LOAD_REQ,
    TR_LOAD_REPLY,
    TR_TID_REQUEST,
    TR_TID_REPLY,
    TR_SKIP,
    TR_NSTID_PROBE_REQ,
    TR_NSTID_PROBE_REPLY,
    TR_MARK,
    TR_COMMIT,
    TR_ABORT,

    // not in table 1
    TR_INVALIDATE,
    TR_INVALIDATE_ACK,

    // new stuff
    TR_OVERFLOW_REQUEST_START,     // core wants permission to overflow x$
    TR_OVERFLOW_STOP,              // tid vendor asks other cores to halt any transactional work
    TR_OVERFLOW_STOP_ACK,          // cores reply they have halted
    TR_OVERFLOW_REQUEST_START_ACK, // tid vendor allows overflow on one core
    TR_OVERFLOW_DONE,              // core finished overflow transaction
    TR_OVERFLOW_RESUME,            // tid vendor allows cores to resume

	CU_TOMMY_1,
	CU_TOMMY_2
};

#define MF_TUP_BEGIN(X) enum X {
#define MF_TUP(X) X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

class tm_manager_inf; 
class Cartographer;
class mem_fetch {
friend class Cartographer;
friend class CartographerMem;
friend class shader_core_ctx;
friend class data_cache;
friend class shader_core_mem_fetch_allocator;
friend class baseline_cache;
friend class l1_cache;
friend class l2_cache;
friend class tx_log_walker;
friend class tx_log_walker_warpc;
friend class tex_cache;
friend class commit_unit;
friend class L2icnt_interface;
friend class gpgpu_sim;
public:
    mem_fetch( const mem_access_t &access, 
               const warp_inst_t *inst,
               unsigned ctrl_size, 
               unsigned wid,
               unsigned sid, 
               unsigned tpc, 
               const class memory_config *config );
   ~mem_fetch();

   void set_status( enum mem_fetch_status status, unsigned long long cycle );
   void set_type( enum mf_type t ) { m_type=t; }
   void set_reply() 
   { 
       assert( m_access.get_type() != L1_WRBK_ACC && m_access.get_type() != L2_WRBK_ACC );
       if( m_type==READ_REQUEST ) {
           assert( !get_is_write() );
           m_type = READ_REPLY;
       } else if( m_type == WRITE_REQUEST ) {
           assert( get_is_write() );
           m_type = WRITE_ACK;
       }
   }
   void do_atomic();

   void print( FILE *fp, bool print_inst = true ) const;

   const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
   unsigned get_data_size() const { return m_data_size; }
   void     set_data_size( unsigned size ) { m_data_size=size; }
   unsigned get_ctrl_size() const { return m_ctrl_size; }
   unsigned size() const { return m_data_size+m_ctrl_size; }
   bool is_write() {return m_access.is_write();}
   void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
   new_addr_type get_addr() const { return m_access.get_addr(); }
   new_addr_type get_partition_addr() const { return m_partition_addr; }
   unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
   bool     get_is_write() const { return m_access.is_write(); }
   unsigned get_request_uid() const { return m_request_uid; }
   unsigned get_sid() const { return m_sid; }
   unsigned get_tpc() const { return m_tpc; }
   unsigned get_wid() const { return m_wid; }
   bool istexture() const;
   bool isconst() const;
   enum mf_type get_type() const { return m_type; }
   bool isatomic() const;

   void set_return_timestamp( unsigned t ) { m_timestamp2=t; }
   void set_icnt_receive_time( unsigned t ) { m_icnt_receive_time=t; }
   unsigned get_timestamp() const { return m_timestamp; }
   unsigned get_return_timestamp() const { return m_timestamp2; }
   unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }

   enum mem_access_type get_access_type() const { return m_access.get_type(); }
   const active_mask_t& get_access_warp_mask() const { return m_access.get_warp_mask(); }
   mem_access_byte_mask_t get_access_byte_mask() const { return m_access.get_byte_mask(); }
   mem_access_byte_mask_t get_byte_mask() const { return m_access.get_byte_mask(); }
   new_addr_type get_fill_addr() const { return m_access.get_fill_addr(); }
   bool is_tx_load() const { return m_access.is_tx_load(); }
   address_type get_pc() const { return m_inst.empty()?-1:m_inst.pc; }
   const warp_inst_t &get_inst() { return m_inst; }
   enum mem_fetch_status get_status() const { return m_status; }

   const memory_config *get_mem_config(){return m_mem_config;}

   unsigned get_num_flits(bool simt_to_mem);

   bool is_transactional() const { return m_transactional; }
   void set_is_transactional() { m_transactional=true; }
   unsigned get_transaction_id() const { return m_transaction_id; }
   void set_transaction_id( unsigned trid ) { m_transaction_id=trid; }
   unsigned get_memory_partition_id() const { return m_raw_addr.chip; }
   void set_memory_partition_id( unsigned chip ) { m_raw_addr.chip=chip; }
   void set_sub_partition_id( unsigned sub_partition ) { m_raw_addr.sub_partition = sub_partition; }
   void set_commit_unit_generated() { m_commit_unit_generated = true; }
   bool get_commit_unit_generated() { return m_commit_unit_generated; }
   void set_commit_id( int cid ) { m_commit_id = cid; }
   int get_commit_id( ) { return m_commit_id; }

   void set_tm_manager_ptr(class tm_manager_inf * p_tm_manager) { m_tm_manager = p_tm_manager; }
   class tm_manager_inf* get_tm_manager_ptr() { return m_tm_manager; }

   void set_commit_pending_ptr(std::bitset<16> *commit_pending) { m_commit_pending_flag = commit_pending; }
   std::bitset<16>* get_commit_pending_ptr() { return m_commit_pending_flag; }

   // interface to allow a single mem_fetch to encompass multiple coalesced mem_fetch 
   bool has_coalesced_packet() const;
   void append_coalesced_packet(mem_fetch *mf); 
   mem_fetch* next_coalesced_packet(); 
   void pop_coalesced_packet(); 
   bool partial_processed_packet() const; // detect arrival of new coalesced packet 
   std::list<mem_fetch*>& get_coalesced_packet_list(); // retrieve the list of coalesced packet for batch processing 

private:
   // request source information
   unsigned m_request_uid;
   unsigned m_last_request_uid; // for debugging dangling pointers 
   unsigned m_sid;
   unsigned m_tpc;
   unsigned m_wid;

   // where is this request now?
   enum mem_fetch_status m_status;
   unsigned long long m_status_change;

   // request type, address, size, mask
   mem_access_t m_access;
   unsigned m_data_size; // how much data is being written
   unsigned m_ctrl_size; // how big would all this meta data be in hardware (does not necessarily match actual size of mem_fetch)
   new_addr_type m_partition_addr; // linear physical address *within* dram partition (partition bank select bits squeezed out)
   addrdec_t m_raw_addr; // raw physical address (i.e., decoded DRAM chip-row-bank-column address)
   enum mf_type m_type;

   // statistics
   unsigned m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
   unsigned m_timestamp2; // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed onto icnt to shader; only used for reads
   unsigned m_icnt_receive_time; // set to gpu_sim_cycle + interconnect_latency when fixed icnt latency mode is enabled

   // requesting instruction (put last so mem_fetch prints nicer in gdb)
   warp_inst_t m_inst;

   // transactional memory request should be treated differently 
   bool m_transactional; 
   unsigned m_transaction_id;

   // commit unit generated 
   bool m_commit_unit_generated; 
   int m_commit_id; 

   // for commit unit side commit 
   class tm_manager_inf* m_tm_manager; 

   std::bitset<16>* m_commit_pending_flag; 

   // list of coalesced mem_fetch in this mem_fetch 
   std::list<mem_fetch*> m_coalesced_packets; 
   bool m_coalesced_popped; 

   static unsigned sm_next_mf_request_uid;

   const class memory_config *m_mem_config;
   unsigned icnt_flit_size;
   int m_serial_number;
   int rct_to_cat_serial_number;
   unsigned long long rct_to_cat_timestamp;
};

#endif

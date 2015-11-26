#ifndef __CARTOGRAPHER_H
#define __CARTOGRAPHER_H

// 2013-07-27: A makeshift idea for expressing data dependency would be appending edges to the critical path graph.

//#include "cuda-sim/instructions.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx.tab.h"
#include "cuda-sim/ptx_sim.h"
#include <stdio.h>
#include <stdlib.h>

#include "cuda-sim/opcodes.h"
#include "cuda-sim/../intersim/statwraper.h"
#include <set>
#include <map>
#include <vector>
#include <deque>
#include <string>
#include "cuda-sim/../abstract_hardware_model.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx-stats.h"
#include "cuda-sim/ptx_loader.h"
#include "cuda-sim/ptx_parser.h"
#include "cuda-sim/ptx_sim.h"
#include "cuda-sim/../gpgpusim_entrypoint.h"
#include "cuda-sim/decuda_pred_table/decuda_pred_table.h"
#include "cuda-sim/../stream_manager.h"
#include "cuda-sim/tm_manager_internal.h"
#include <sstream>
#include <iostream>
#include "gpgpu-sim/mem_fetch.h"
#include "gpgpu-sim/shader.h"
#include "gpgpu-sim/traffic_breakdown.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/l2cache.h"

#include "sqlite3.h"
#include <zlib.h>

extern unsigned long long  gpu_tot_sim_cycle, gpu_sim_cycle;

extern void updateInstSerailID(inst_t* inst, long sn);
class function_info;
class shader_core_ctx;
class Cartographer;
class CartographerTimeSeries;

static std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems) {
	if(s.length() < 0) return elems;
	std::stringstream ss(s);
	std::string item;
	while(std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}



// What other components is 1 instruction related to ?
//
// 1 instance of InstInPipeline represents 1 instruction in one warp.
// That means, it's executed on 32 threads.
// Moved to .cc
class InstInPipeline;

typedef struct MaskState {
	unsigned long the_mask;
	unsigned char count;
} MaskState;

typedef enum MyFuncUnit {
	MY_SP,
	MY_SFU,
	MY_MEM
} MyFuncUnit;

// Grand-level events
typedef enum MyTxnEvent {
	MY_TXN_BEGIN,
	MY_TXN_ROLLBACK,
	MY_TXN_COMMIT
} MyTxnEvent;

// Just after fetching = 0
// Just after decoding = 1
// Just before demise = 99
extern long ilti_serial_id;
class InstLifeTimeInfo {
private:
	void init() {
		inst = NULL; inst_string = ""; pc = 0xBAADCAFE;
		fetch_began = icache_access_ready = scheduled = executed = retired = 0;
		func_unit = NULL; notes = stage = sn = sched_num_issued_constraint_until = 0;
		is_fetch_missed = false; ldst_access_ready = 0;
		scoreboard_deps.clear(); mem_stalledby.clear(); sfu_stalledby.clear();
		serial = -2147483648;
		active_mask.reset();
	}
public:
	InstLifeTimeInfo(int _sn) {
		init();
		sn = _sn;
		serial = (ilti_serial_id++);
		active_mask.reset();
	}
	InstLifeTimeInfo() { init(); }
	InstLifeTimeInfo(sqlite3_stmt* row) {
		is_tcommit = false;
		inst = NULL;
		shader_id = sqlite3_column_int(row, 0);
		warp_id = sqlite3_column_int(row, 1);
		inst_string = "";
		const unsigned char* c = sqlite3_column_text(row, 2);
		inst_string = (const char*)c;
		fetch_began = sqlite3_column_int64(row, 3);
		icache_access_ready = sqlite3_column_int64(row, 4);
		scheduled   = sqlite3_column_int64(row, 5);
		sched_num_issued_constraint_until = sqlite3_column_int64(row, 6);
		stalled_in_fu_until = sqlite3_column_int64(row, 7);
		executed = sqlite3_column_int64(row, 8);
		active_mask = active_mask_t(sqlite3_column_int64(row, 9));
		retired  = sqlite3_column_int64(row, 10);
		func_unit = (char*)sqlite3_column_text(row, 11);
		serial = sqlite3_column_int(row, 12);
		is_fetch_missed = (fetch_began < icache_access_ready);
		c = sqlite3_column_text(row, 13);
		std::string sbdeps = (const char*)c;
		deserializeScoreboardDeps(sbdeps);

		c = sqlite3_column_text(row, 14);
		std::string fustalled = (const char*)c;
		deserializeFUStalledBys(fustalled);
		this->ldst_access_ready = sqlite3_column_int64(row, 15);
	}
	InstLifeTimeInfo(const InstLifeTimeInfo& other) {
		inst = other.inst;
		inst_string = other.inst_string;
		fetch_began = other.fetch_began;
		icache_access_ready = other.icache_access_ready;
		scheduled = other.scheduled;
		func_unit = other.func_unit;
		retired = other.retired;
		executed = other.executed;
		notes = other.notes;
		stage = other.stage;
		sn = other.sn;
		pc = other.pc;
		is_fetch_missed = other.is_fetch_missed;
		sched_num_issued_constraint_until = other.sched_num_issued_constraint_until;
		stalled_in_fu_until = other.stalled_in_fu_until;
		scoreboard_deps = other.scoreboard_deps;
		mem_stalledby = other.mem_stalledby;
		sp_stalledby = other.sp_stalledby;
		sfu_stalledby = other.sfu_stalledby;
//		serial = (ilti_serial_id++); // 2 issues so need new sn
		serial = other.serial;
		updateInstSerailID(inst, serial);
		ldst_access_ready = other.ldst_access_ready;
		is_tcommit = other.is_tcommit;
		warp_id = other.warp_id;
		shader_id = other.shader_id;
		active_mask = other.active_mask;
		assert (!(other.ldst_access_ready > 1e9));
		// Constraint
	}
	bool sanityCheck() {
		return ((sched_num_issued_constraint_until <= scheduled) &&
		         (fetch_began <= icache_access_ready) &&
		         (stalled_in_fu_until <= executed));
	}
	void getNewSerialID() {
		serial = (ilti_serial_id++);
	}
	bool operator==(const InstLifeTimeInfo& other) {
		return (inst == other.inst &&
			inst_string == other.inst_string &&
			fetch_began == other.fetch_began &&
			icache_access_ready == other.icache_access_ready &&
			scheduled == other.scheduled &&
			sched_num_issued_constraint_until == other.sched_num_issued_constraint_until &&
			executed == other.executed &&
			func_unit == other.func_unit &&
			retired == other.retired &&
			notes == other.notes &&
			stage == other.stage &&
			sn == other.sn &&
			pc == other.pc &&
			is_fetch_missed == other.is_fetch_missed &&
			stalled_in_fu_until == other.stalled_in_fu_until &&
			ldst_access_ready == other.ldst_access_ready &&
			is_tcommit == other.is_tcommit &&
			active_mask == other.active_mask
		);
	}
	void print(FILE* f) {
		fprintf(f, "[%s](%p) PC=%X FB=%llu I$Ready=%u Sch=%llu SchC=%llu Ex=%llu Ret=%llu FU=%s StallFU=%llu Note=%u St=%u SN=%d\n",
			inst_string.c_str(),
			inst, pc,
			fetch_began, icache_access_ready, scheduled, sched_num_issued_constraint_until, executed, retired, func_unit,
			stalled_in_fu_until,
			notes, stage, sn
		);
	}
	void print_debug(FILE* f) {
		print(f);
		inst->print_insn(f);
		inst->print(f);
		fprintf(f, "\n");
	}

	std::string serializeScoreboardDeps();
	void deserializeScoreboardDeps(std::string str);
	std::string serializeFUStalledBys();
	void deserializeFUStalledBys(std::string str);

	bool insertToSQLiteDB(sqlite3* the_db, unsigned shader_id, unsigned warp_id) {
		if(!the_db) {
			fprintf(stderr, "DB is null\n");
			return false;
		}
		sqlite3_stmt* insert_stmt;
		int err;
		std::string insert_query = "INSERT INTO sandwich (CoreID, WarpID, Instruction, Fetch, ICacheAccessReady, Scheduled,";
		insert_query += " SchConstraintDelay, FUStalled, Executed, ActiveMask, Retired, FU, SN, ScoreboardDeps, FUStalledBy,";
		insert_query += " LDSTAccessReady) ";
		insert_query += " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

		err = sqlite3_prepare_v2(the_db, insert_query.c_str(), -1, &insert_stmt, NULL);
		if(err != SQLITE_OK) { printf("Err = %d\n", err); exit(1); }

		err = sqlite3_bind_int(insert_stmt, 1, shader_id); // Shader ID
		if(err != SQLITE_OK) { printf("(1) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 2, warp_id); // Warp ID
		if(err != SQLITE_OK) { printf("(2) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_text(insert_stmt, 3, inst_string.c_str(), -1, SQLITE_TRANSIENT);
		if(err != SQLITE_OK) { printf("(3) In binding text err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 4, fetch_began); // Fetch
		if(err != SQLITE_OK) { printf("(4) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 5, icache_access_ready); // FetchDeltaT
		if(err != SQLITE_OK) { printf("(5) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 6, scheduled); // Scheduled
		if(err != SQLITE_OK) { printf("(6) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 7, sched_num_issued_constraint_until); // SchConstraintDelay
		if(err != SQLITE_OK) { printf("(7) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int64(insert_stmt, 8, stalled_in_fu_until); // FUStalled
		if(err != SQLITE_OK) { printf("(8) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int64(insert_stmt, 9, executed); // Executed
		if(err != SQLITE_OK) { printf("(9) In binding int err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 10, active_mask.to_ulong());
		if (err != SQLITE_OK) { printf("(10) In binding int64 err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 11, retired); // Retired
		if(err != SQLITE_OK) { printf("(11) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_text(insert_stmt, 12, (const char*)func_unit, -1, SQLITE_TRANSIENT); // FU
		if(err != SQLITE_OK) { printf("(12) In binding text err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 13, serial); // Serial Number (should use serial not sn)
		if(err != SQLITE_OK) { printf("(13) In binding int err = %d\n", err); exit(1); }

		std::string sb_deps = this->serializeScoreboardDeps();
		err = sqlite3_bind_text(insert_stmt, 14, sb_deps.c_str(), -1, SQLITE_TRANSIENT); // ScoreboardDeps
		if(err != SQLITE_OK) { printf("(14)In binding text err = %d\n", err); exit(1); }

		std::string fu_stalled = this->serializeFUStalledBys();
		err = sqlite3_bind_text(insert_stmt, 15, fu_stalled.c_str(), -1, SQLITE_TRANSIENT); // FUStalledBy
		if(err != SQLITE_OK) { printf("(15)In binding text err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 16, ldst_access_ready);
		if(err != SQLITE_OK) { printf("(16) In binding int err = %d\n", err); exit(1); } // LDST Access Ready

		err = sqlite3_step(insert_stmt);
		if(err != SQLITE_DONE) {
			std::cerr << "Can't insert!\n";
			exit(1);
		}

		sqlite3_finalize(insert_stmt);
		// Data corruption error !?
		assert (ldst_access_ready < 1e9);
		return true;
	}

	unsigned char warp_id, shader_id;
	warp_inst_t* inst;
	std::string inst_string;
	address_type pc;
	unsigned long long fetch_began;
	unsigned int icache_access_ready;
	unsigned long long scheduled;// Sent to FU's in port; (== issue warp), in scheduler_unit::cycle()
	unsigned long long executed; // Starts execution; (== issue to FU), in shader_core_ctx::execute()
	unsigned long long retired;
	char* func_unit;
	unsigned char notes;
	unsigned char stage; // After fetch, awaiting decode
	int sn;

	unsigned long long sched_num_issued_constraint_until; // Blocked until this cycle b/c scheduler #issue limit
	unsigned long long stalled_in_fu_until; // Stalled in FU until this clock cycle

	bool is_fetch_missed; // If this is true then this warp fetches the inst for everybody else

	std::set<long> scoreboard_deps; // Serial ID
	// 2013-07-26: stalled by which guys?
	std::set<long> sp_stalledby;
	std::set<long> sfu_stalledby;
	std::set<long> mem_stalledby;
	long serial;
	unsigned long long ldst_access_ready;

	bool is_tcommit;
	active_mask_t active_mask;
};

// Transaction state machine transition timestamps
class MyCommittingTxnEntry {
public:
	MyCommittingTxnEntry(shader_core_ctx* _core) : core (_core) {
		wid = cid = tid_in_warp = sandwich_sn = (unsigned) (-1);
		has_done_fill = false;
		send_rs_started = send_ws_started = done_fill = send_tx_passfail = done_scoreboard_commit = (unsigned)(-1);
	}
	MyCommittingTxnEntry(sqlite3_stmt* row) {
		sandwich_sn = sqlite3_column_int(row, 0);
		tid_in_warp = sqlite3_column_int(row, 1);
		send_rs_started = sqlite3_column_int64(row, 2);
		send_ws_started = sqlite3_column_int64(row, 3);
		done_fill = sqlite3_column_int64(row, 4);
		send_tx_passfail = sqlite3_column_int64(row, 5);
		done_scoreboard_commit = sqlite3_column_int64(row, 6);
		std::string rs_sz((const char*)sqlite3_column_text(row, 7));
		rs_etys_time = deserializeArray(rs_sz);
		std::string ws_sz((const char*)sqlite3_column_text(row, 8));
		ws_etys_time = deserializeArray(ws_sz);
		std::string cureply_sz = ((const char*)sqlite3_column_text(row, 9));
		cu_replies_time = deserializeArray(cureply_sz);
	}
	void sanityCheck() {
		assert (tid_in_warp >= 0 and tid_in_warp <= 31);
	}
	shader_core_ctx* core;
	struct tx_log_walker::warp_commit_tx_t commit_warp_info;
	unsigned wid, cid; // Warp ID and Commit ID
	unsigned tid_in_warp;
	bool has_done_fill;
	std::vector<unsigned long long> rs_etys_time, ws_etys_time, cu_replies_time;
	unsigned long long send_rs_started, // Start sending RS
	                      send_ws_started,
						  done_fill,       // Send to ALL CUs
						  send_tx_passfail,
						  done_scoreboard_commit;
	unsigned sandwich_sn;
	bool insertToSQLiteDB(sqlite3* the_db) {
		if (!the_db) { fprintf(stderr, "DB is NULL\n"); return false; }
		sqlite3_stmt* insert_stmt;
		int err;

		// Insert-time sanity check!

		const char* insert_query = "INSERT INTO sandwich_commit_tx "
			"(SN, TIDInWarp, SendRS, SendWS, DoneFill, SendTxPassFail, DoneScoreboardCommit, "
			"RSDetails, WSDetails, CUReplyDetails) "
			"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

		err = sqlite3_prepare_v2(the_db, insert_query, -1, &insert_stmt, NULL);
		if (err != SQLITE_OK) { printf("Err = %d\n", err); exit(1); }

		err = sqlite3_bind_int(insert_stmt, 1, sandwich_sn);
		if (err != SQLITE_OK) { printf("(1) in binding int err = %d\n", err); exit(1); }

		err = sqlite3_bind_int(insert_stmt, 2, tid_in_warp);
		if (err != SQLITE_OK) { printf("(2) in binding int err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 3, send_rs_started);
		if (err != SQLITE_OK) { printf("(3) in binding int64 err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 4, send_ws_started);
		if (err != SQLITE_OK) { printf("(4) in binding int64 err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 5, done_fill);
		if (err != SQLITE_OK) { printf("(5) in binding int64 err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 6, send_tx_passfail);
		if (err != SQLITE_OK) { printf("(6) in binding int64 err = %d\n", err); exit(1); }

		err = sqlite3_bind_int64(insert_stmt, 7, done_scoreboard_commit);
		if (err != SQLITE_OK) { printf("(7) in binding int64 err = %d\n", err); exit(1); }

		std::string rs_sz = getRWSetDetailString('R');
		err = sqlite3_bind_text(insert_stmt, 8, rs_sz.c_str(), -1, SQLITE_TRANSIENT);
		if (err != SQLITE_OK) { printf("(8) in binding text err = %d\n", err); exit(1); }

		std::string ws_sz = getRWSetDetailString('W');
		err = sqlite3_bind_text(insert_stmt, 9, ws_sz.c_str(), -1, SQLITE_TRANSIENT);
		if (err != SQLITE_OK) { printf("(9) in binding text err = %d\n", err); exit(1); }

		std::string cur_sz = getRWSetDetailString('D');
		err = sqlite3_bind_text(insert_stmt, 10, cur_sz.c_str(), -1, SQLITE_TRANSIENT);
		if (err != SQLITE_OK) { printf("(10) in binding text err = %d\n", err); exit(1); }

		err = sqlite3_step(insert_stmt);
		if (err != SQLITE_DONE) { printf("in Step err = %d\n", err); exit(1); }

		sqlite3_finalize(insert_stmt);

		return true;
	}
private:
	std::string getRWSetDetailString(char rw) {
		std::string ret = "";
		std::vector<unsigned long long>* ptr = (rw == 'R' ? &rs_etys_time :
				(rw == 'W' ? &ws_etys_time : (rw == 'D' ? &cu_replies_time : NULL )));
		assert (ptr);
		for (unsigned i=0; i<ptr->size(); i++) {
			if (i > 0) ret = ret + ",";
			ret = ret + std::to_string(ptr->at(i));
		}
		return ret;
	}
	std::vector<unsigned long long> deserializeArray(const std::string& sz) {
		std::vector<unsigned long long> ret;
		std::vector<std::string> tmp;
		split(sz, ',', tmp);
		for(std::vector<std::string>::iterator itr = tmp.begin();
				itr != tmp.end(); itr++) {
			std::string sz = *itr;
			unsigned long long k = atol(sz.c_str());
			ret.push_back(k);
		}
		return ret;
	}
};

class MyMemFetchInfo {
public:
	short warp_id, shader_id;
	unsigned long long sent_to_icnt, received_from_icnt;
	int serial_number;
	mem_fetch* mf;
	bool operator< (const MyMemFetchInfo& other) const {
		if (shader_id < other.shader_id) return true;
		else {
			if (warp_id < other.warp_id) return true;
			else {
				if (sent_to_icnt < other.sent_to_icnt) return true;
				else {
					if (received_from_icnt < other.received_from_icnt) return true;
					else {
						if (serial_number < other.serial_number) return true;
						else {
							if ((unsigned long long)mf < (unsigned long long)(other.mf)) return true;
							else return false;
						}
					}
				}
			}
		}
		return false;
	}
};

typedef std::pair<unsigned, unsigned> SHADER_WARP_ID_Ty;

class CoalescedMemoryAccessInfo {
public:
	unsigned shader_id;
	unsigned warp_id;
	unsigned dynamic_warp_id;
	unsigned subwarp_id;
	unsigned long long cycle;
	new_addr_type address;
	unsigned block_address;
	unsigned chunk;

	void print(FILE* f) {
		fprintf(f, "Shader %u, Warp %u, DynWarp %u, Subwarp %u, cycle %llu, ADDR=%p, BlockAddr=%p, chunk=%u\n",
			shader_id, warp_id, dynamic_warp_id, subwarp_id, cycle, (void*)address, (void*)((new_addr_type)(0x00000000FFFFFFFF & block_address)), chunk);
	}

	CoalescedMemoryAccessInfo(const SHADER_WARP_ID_Ty& swid, unsigned _active_warp_id, unsigned _subwarp_id,
			unsigned long long _cycle, new_addr_type addr,
		unsigned blkaddr, unsigned chk) : shader_id(swid.first), warp_id(swid.second), dynamic_warp_id(_active_warp_id),
				subwarp_id(_subwarp_id), cycle(_cycle),
			address(addr), block_address(blkaddr), chunk(chk) {}

	bool insertToSQLiteDB(sqlite3* the_db) {
		if(!the_db) {
			fprintf(stderr, "DB is null\n");
			return false;
		}
		sqlite3_stmt* insert_stmt;
		int err;
		std::string insert_query = "INSERT INTO koalesced (CoreID, WarpID, DynamicWarpID, SubwarpID, Cycle, Addr,";
		insert_query += " BlockAddr, Chunk) ";
		insert_query += " VALUES (?, ?, ?, ?, ?, ?, ?, ?);";

		err = sqlite3_prepare_v2(the_db, insert_query.c_str(), -1, &insert_stmt, NULL);
		if(err != SQLITE_OK) { printf("Err = %d\n", err); exit(1); }

		err = sqlite3_bind_int(insert_stmt, 1, shader_id); // Shader ID
		if(err != SQLITE_OK) { printf("(1) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 2, warp_id); // Warp ID
		if(err != SQLITE_OK) { printf("(2) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 3, dynamic_warp_id);
		if(err != SQLITE_OK) { printf("(3) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 4, subwarp_id);
		if(err != SQLITE_OK) { printf("(4) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int64(insert_stmt, 5, cycle);
		if(err != SQLITE_OK) { printf("(5) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int64(insert_stmt, 6, address);
		if(err != SQLITE_OK) { printf("(6) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int64(insert_stmt, 7, block_address);
		if(err != SQLITE_OK) { printf("(7) In binding int err = %d\n", err); exit(1); }
		err = sqlite3_bind_int(insert_stmt, 8, chunk);
		if(err != SQLITE_OK) { printf("(8) In binding int err = %d\n", err); exit(1); }

		err = sqlite3_step(insert_stmt);
		if(err != SQLITE_DONE) {
			std::cerr << "Can't insert!\n";
			exit(1);
		}
		sqlite3_finalize(insert_stmt);

		return true;
	}

	CoalescedMemoryAccessInfo(sqlite3_stmt* row) {
		shader_id = sqlite3_column_int(row, 0);
		warp_id = sqlite3_column_int(row, 1);
		dynamic_warp_id = sqlite3_column_int(row, 2);
		subwarp_id = sqlite3_column_int(row, 3);
		cycle = sqlite3_column_int64(row, 4);
		address = sqlite3_column_int64(row, 5);
		block_address = sqlite3_column_int64(row, 6);
		chunk = sqlite3_column_int(row, 7);
	}
};

// SIMT Stack state changes
typedef std::list<std::pair<struct MaskState, unsigned> > * MASK_HIST_ENTRY_P_Ty;
typedef std::map<SHADER_WARP_ID_Ty, MASK_HIST_ENTRY_P_Ty> MASK_HIST_Ty;
typedef std::list<unsigned long long>* MASK_TIME_ENTRY_P_Ty;
typedef std::map<SHADER_WARP_ID_Ty, MASK_TIME_ENTRY_P_Ty> MASK_TIME_Ty;

// Warp Scheduled Event (When is a warp scheduled?
typedef std::pair<unsigned long long, char> WARP_SCHEDULED_TIME_ENTRY;
typedef std::list<WARP_SCHEDULED_TIME_ENTRY>* WARP_SCHEDULED_TIME_ENTRY_LIST_P_Ty;
typedef std::map<SHADER_WARP_ID_Ty, WARP_SCHEDULED_TIME_ENTRY_LIST_P_Ty> WARP_SCHEDULED_TIME_Ty;

// Whitelist-related
typedef std::pair<unsigned long long, unsigned long long> CYCLE_INTERVAL_Ty;
typedef std::list<CYCLE_INTERVAL_Ty> CYCLE_INTERVAL_LIST_Ty;

class TxnEpochStates {
public:
	char outcome; // A = abort, C = commit
	unsigned uid, epoch, hw_sid, hw_wid, hw_tid;
	long cycle_start, cycle_send_rw, cycle_wait_cu_reply, cycle_end;
	dim3 ctaid, tid;
	unsigned read_set_size,  read_set_size_nz; // nz = non zero; These sizes are obtained by counting
	unsigned write_set_size, write_set_size_nz;
	unsigned n_read, n_read_all, n_write; // BW data, as recorded by tm_manager
	unsigned read_set_size_1, write_set_size_1; // Word-level RS/WS Sizes
	unsigned read_set_size_2, write_set_size_2; // These sizes are struct tm_warp_info::m_{read,write}_log_size.
	addr_set_t read_word_set, write_word_set;
	TxnEpochStates() :
		read_set_size(0), read_set_size_nz(0), write_set_size(0), write_set_size_nz(0),
		n_read(0), n_read_all(0), n_write(0), read_set_size_1(0), write_set_size_1(0),
		cycle_start(-1), cycle_send_rw(-1), cycle_wait_cu_reply(-1), cycle_end(-1),
		read_set_size_2(0), write_set_size_2(0)
	{}
	void appendToGzFile(gzFile f) {
		gzwrite(f, &outcome, sizeof(outcome));
		gzwrite(f, &uid,   sizeof(uid));
		gzwrite(f, &epoch, sizeof(epoch));
		gzwrite(f, &hw_sid, sizeof(hw_sid));
		gzwrite(f, &hw_wid, sizeof(hw_wid));
		gzwrite(f, &hw_tid, sizeof(hw_tid));

		gzwrite(f, &ctaid, sizeof(ctaid));
		gzwrite(f, &tid,  sizeof(tid));

		gzwrite(f, &cycle_start, sizeof(cycle_start));
		gzwrite(f, &cycle_send_rw, sizeof(cycle_send_rw));
		gzwrite(f, &cycle_wait_cu_reply, sizeof(cycle_wait_cu_reply));
		gzwrite(f, &cycle_end, sizeof(cycle_end));

		gzwrite(f, &read_set_size, sizeof(read_set_size));
		gzwrite(f, &read_set_size_nz, sizeof(read_set_size_nz));
		gzwrite(f, &write_set_size, sizeof(write_set_size));
		gzwrite(f, &write_set_size_nz, sizeof(write_set_size_nz));
		gzwrite(f, &n_read, sizeof(n_read));
		gzwrite(f, &n_read_all, sizeof(n_read_all));
		gzwrite(f, &n_write, sizeof(n_write));
		gzwrite(f, &read_set_size_1, sizeof(read_set_size_1));
		gzwrite(f, &write_set_size_1, sizeof(write_set_size_1));
		gzwrite(f, &read_set_size_2, sizeof(read_set_size_2));
		gzwrite(f, &write_set_size_2, sizeof(write_set_size_2));
		for (addr_set_t::iterator itr = read_word_set.begin(); itr != read_word_set.end(); itr++) {
			unsigned addr = *itr;
			gzwrite(f, &addr, sizeof(addr));
		}
		for (addr_set_t::iterator itr = write_word_set.begin(); itr != write_word_set.end(); itr++) {
			unsigned addr = *itr;
			gzwrite(f, &addr, sizeof(addr));
		}
	}
	void readFromGzFile(gzFile f) {
		gzread(f, &outcome, sizeof(outcome));
		gzread(f, &uid, sizeof(uid));
		gzread(f, &epoch, sizeof(epoch));
		gzread(f, &hw_sid, sizeof(hw_sid));
		gzread(f, &hw_wid, sizeof(hw_wid));
		gzread(f, &hw_tid, sizeof(hw_tid));

		gzread(f, &ctaid, sizeof(ctaid));
		gzread(f, &tid,   sizeof(tid));

		gzread(f, &cycle_start, sizeof(cycle_start));
		gzread(f, &cycle_send_rw, sizeof(cycle_send_rw));
		gzread(f, &cycle_wait_cu_reply, sizeof(cycle_wait_cu_reply));
		gzread(f, &cycle_end, sizeof(cycle_end));

		gzread(f, &read_set_size, sizeof(read_set_size));
		gzread(f, &read_set_size_nz, sizeof(read_set_size_nz));
		gzread(f, &write_set_size, sizeof(write_set_size));
		gzread(f, &write_set_size_nz, sizeof(write_set_size_nz));
		gzread(f, &n_read, sizeof(n_read));
		gzread(f, &n_read_all, sizeof(n_read_all));
		gzread(f, &n_write, sizeof(n_write));
		gzread(f, &read_set_size_1, sizeof(read_set_size_1));
		gzread(f, &write_set_size_1, sizeof(write_set_size_1));
		gzread(f, &read_set_size_2, sizeof(read_set_size_2));
		gzread(f, &write_set_size_2, sizeof(write_set_size_2));
		read_word_set.clear();
		for (int i=0; i<read_set_size_1; i++) {
			unsigned addr;
			gzread(f, &addr, sizeof(addr));
			read_word_set.insert(addr);
		}
		write_word_set.clear();
		for (int i=0; i<write_set_size_1; i++) {
			unsigned addr;
			gzread(f, &addr, sizeof(addr));
			write_word_set.insert(addr);
		}
	}
};

class Cartographer {
public:
	Cartographer() : logfile(stdout), is_log_barrier(true), is_using_whitelist(false),
		shaders_logged(), cycle_segments_logged(),
		activity_onoff(),
		l2_read_miss(), l2_write_miss(),
		is_log_simt_states(false),
		is_log_warp_sched_evts(false),
		is_log_sandwich_plot(false), num_warps_per_shader(0), is_log_simdfu_occupancy(false) {
		m_l2miss_hist_interval = 1000;
		for(unsigned i=0; i<42; i++) { // Assume 42 shaders at most
			edges_fetch.push_back(std::map<unsigned, InstLifeTimeInfo>());
		}
		curr_kernel_id = -1;
		m_live_tcommit_serial.clear();
	}
	void PrintDummyFunctionInfoGraph(function_info* finfo, FILE* fh);

	// Tracing a single instruction
	// What components of the G.P.U. has this instruction gone through?
	// These should become the sites where we the nodes in the final graph are marked.
	//
	// Question: Is there a way to gain access to the CONTEXT where I want to insert this line of code?
	//           Is this called a closure ?
	//
	void On_shader_core_fetch_L1I_miss(shader_core_ctx* from, mem_fetch* mf);
	void On_shader_core_fetch_L1I_hit(shader_core_ctx* from, mem_fetch* mf);
	void On_shader_core_fetch_L1I_access(shader_core_ctx* from, mem_fetch* mf);
	void On_shader_core_decode(shader_core_ctx* from, const warp_inst_t* inst);


	FILE* logfile;
	// Log barrier information.
	bool is_log_barrier;
	unsigned log_barrier_coreid; // We should log all CTAs that have been executed on this core.

	// Whitelist-related
	bool is_using_whitelist;
	std::set<unsigned> shaders_logged;
	CYCLE_INTERVAL_LIST_Ty cycle_segments_logged;
	CYCLE_INTERVAL_LIST_Ty::iterator curr_whitelist_cycle_seg_itr;

	// Lemme have some logging function.
	void onIssueBlockToCore(shader_core_ctx* from, unsigned cta_hw_id,
			unsigned warp_id_min, unsigned warp_id_max,
			kernel_info_t &kernel);
	void onCoreFinishesBlock(shader_core_ctx* from, unsigned cta_hw_id);
	void onCoreFinishesWarp(shader_core_ctx* from, unsigned warp_id);
	void onWarpReachesBarrier(shader_core_ctx* core, unsigned warp_id);
	void onAllWarpsReachBarrier(shader_core_ctx* core);
	std::map<unsigned, std::vector<std::vector<std::pair<unsigned long long, unsigned long long> > > > activity_onoff;
	void writeWarpsOnOffHistory(char* file_name);

	// 06-15, Tracing L2 Cache Misses
	void onL2ClockCycleUpdateMissCount(unsigned mem_partition_idx, unsigned delta_read_miss, unsigned delta_write_miss);
	std::map<unsigned, unsigned> l2_read_miss;
	std::map<unsigned, unsigned> l2_write_miss;
	void summary();

	std::map<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> > m_l2_misses; // Current L2 Misses: Key is [SID, WID], Value is #miss
	std::map<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> > m_l1_misses; // Current L1 Misses: Key is [SID, WID], Value is #miss

	int m_l2miss_hist_interval;
	int m_l1miss_hist_interval;

	void onCacheMissPushedIntoQueue(mem_fetch* mf, const char* lvl, bool is_write);

	// Monitoring SIMT stack state change
	bool is_log_simt_states;
	MASK_HIST_Ty m_masks; // Activation mask in RLE encoding
	MASK_TIME_Ty m_mask_timestamps; // Activation mask RLEs' beginning time
	void logWarpEvent(shader_core_ctx* core_id, unsigned warp_id, const simt_mask_t& top_mask);
	void writeSimtStackHistory(char* file_name);

	// Monitoring when is a warp scheduled
	bool is_log_warp_sched_evts;
	WARP_SCHEDULED_TIME_Ty m_warp_scheduled;
	void onWarpScheduled(shader_core_ctx* shader, unsigned warp_id, char fu);// fu = SFU(0) | SP(1) | MEM(2)
	void writeWarpScheduledHistory(char* file_name);

	// 2013-07-04 Instruction Lifetime Plot
	bool is_log_sandwich_plot;
	std::map<SHADER_WARP_ID_Ty, std::list<InstLifeTimeInfo>* > m_live_inst_lifetimes; // Need random access.
	std::map<SHADER_WARP_ID_Ty, std::list<InstLifeTimeInfo>* > m_done_inst_lifetimes;
	std::map<SHADER_WARP_ID_Ty, std::list<std::pair<warp_inst_t*, int> > > m_live_inst_pipestages; // 0 = after fetch, awaiting decode; 1 = after decode, awaiting opr, etc
	void  onNewInstructionBorn(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf, bool is_missed); // Is missed then ...
	void  onICacheAccessReady(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf);
	void  onInstructionsDecoded(shader_core_ctx* shader, unsigned warp_id, warp_inst_t* inst1, warp_inst_t* inst2); // mf is deleted at this point
	void  onInstructionIssued(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst, warp_inst_t* d_inst, const char* fu,
			const active_mask_t* amask); // Modifies the SN of the Dyn Inst!
	void  onInstructionIssueCountConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst, const std::set<warp_inst_t*>& deps);
	// 07-07
	void  onInstructionIssuedToFU(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst, const warp_inst_t* to_inst); // from "static" to "dynamic"
	void  onInstructionFUConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst);
	// 2015-08-04
	// This is for helping with Scheme 2.
	void onDoneTMCommit(shader_core_ctx* shader, unsigned warp_id);

	// 2015-06-29
	void  onLDSTAccessReady(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf);

	void  onInstructionDemises(shader_core_ctx* shader, unsigned warp_id, warp_inst_t* inst, unsigned char infoidx);
	int  getNextFreeSerialNumberAllWarps(shader_core_ctx* shader);
	int  getNextFreeSerialNumberOneWarp(SHADER_WARP_ID_Ty swid);
	InstLifeTimeInfo* findWarpInstAtPipelineStage(SHADER_WARP_ID_Ty& swid, unsigned char stage);
	InstLifeTimeInfo* findWarpInstAtPipelineStageEX(SHADER_WARP_ID_Ty& swid, const warp_inst_t* _inst, unsigned char stage);
	// 07-26
	InstLifeTimeInfo* findWarpInstAtPipelineStageEXAllWarps(unsigned shader_id, const warp_inst_t* _inst, unsigned char stage);
	void clearAllLiveInstructionLog(shader_core_ctx* shader); // Called when a new kernel is launched or on finishing simulation; need shader for #W/C
	void clearLiveInstructionLog(shader_core_ctx* shader, unsigned warp_id);
	void printLiveInstStats(SHADER_WARP_ID_Ty& swid);
	void printLiveInstStats1(unsigned sid, unsigned wid);
	void writeArchivedInstLifetimeStats(const char* filename);
	void writeArchiveInstLifetimeStatsTxt(const char* filename);
	// 07-08 MoveWarp? What does it do ?
	void onMoveWarp(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from, const warp_inst_t* to, unsigned char stage);

	// edges in critical path
	std::vector<std::map<unsigned, InstLifeTimeInfo> > edges_fetch; // Key: PC; Value: Decoded inst record (only fetch and fetch delta are relevant)
	std::map<const inst_t*, long> ldst_serials; // LD/ST leaves the pipeline very early, so we have to catch their ID's

	// 07-26
	void onInstructionIssueResourceConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst,
		std::vector<warp_inst_t*>* mem, std::vector<warp_inst_t*>* sp, std::vector<warp_inst_t*>* sfu,
		bool memfree, bool spfree, bool sfufree
		);

	// 08-15
	std::list<CoalescedMemoryAccessInfo> coalesced_accesses;
	SHADER_WARP_ID_Ty coalesced_curr_swid; // warp id
	unsigned coalesced_curr_dynamic_warpid; // active warp id
	void setShaderAndWarpID(unsigned shader_id, unsigned warp_id, unsigned active_warp_id); // Called at issue()
	void logCoalescingMemoryAccess(unsigned subwarp_id, new_addr_type addr, unsigned block, unsigned chunk); // Called inside functional simulation
	void openAndInitializeCoalescedDB();
	void dumpCurrentCoalescedToDB(); // So that we will not run out of memory.
	bool shouldLog(unsigned shader_id, unsigned warp_id);

	CartographerTimeSeries* linked_cartographerts;
	void linkCommittingTxnToWarp(SHADER_WARP_ID_Ty swid, MyCommittingTxnEntry* cety);

	// 2015-09-02: Last round of data-gathering prior to submitting to HPCA16
	void onTMManagerStart(unsigned uid);
	void onTMManagerAbort(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi);
	void onTMManagerCommit(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi);
	void onTMManagerSendRS(unsigned uid);       // When the state has switched TO Acq CID
	void onTMManagerWaitCUReply(unsigned uid); // When the state has switched TO WaitCUReply

	void onTMManagerSendRWSetEntry(unsigned uid, char rw, unsigned addr_tag);

	void DumpTMHistory();

	struct FUIdentifier {
		char tag[10];
		int sid, fu_idx;
	};
	struct FUOccupancyState {
		std::bitset<32>                    lane_occupied;
		std::vector<unsigned long long> last_update_cycles, last_update_ticks;
		std::vector<pow2_histogram>        vacant_cycles_histogram, vacant_ticks_histogram;
		std::vector<pow2sum_histogram>     vacant_cycles_sum_histogram, vacant_ticks_sum_histogram;
		unsigned long long curr_tick;
		std::vector<unsigned long long> active_ticks, inactive_ticks; // To get cycle, you need to divide by MULTIPLIER
		void update(const std::bitset<32>& occupied);
	};

	// 2015-10-17: Record Dark Silicon Opportunity
	bool is_log_simdfu_occupancy;
	void recordDarkSiliconOpportunity(simd_function_unit* fu, int sid, int fu_idx, const char* tag);
	void dumpDarkSiliconOpportunity();

private:
	void unlinkCommittingThreadToWarp();

	std::list<InstInPipeline*> tracked_insts;
	void setWarpActivityStatus(shader_core_ctx* core_id, unsigned warp_id, bool is_active);
	void setAllWarpsActivityStatus(shader_core_ctx* core_id, bool is_active);

	bool shouldLog(unsigned shader_id, unsigned warp_id, unsigned long long cycle);

	std::map<SHADER_WARP_ID_Ty, int> curr_max_sn;

	unsigned num_warps_per_shader;
	int curr_kernel_id;

	// 2015-08-04: Link ILTI's SN to Committing Threads
	std::map<SHADER_WARP_ID_Ty, unsigned> m_live_tcommit_serial;
	// 2015-08-12: To debug this thing
	std::set<unsigned> tcommit_issued_serials;
	std::map<unsigned, std::set<MyCommittingTxnEntry*> > m_linked_commit_txns;

	// 2015-09-02: Last round of data-gathering prior to submitting to HPCA16
	std::map<unsigned, std::vector<std::pair<char, unsigned> > > uid_to_timestamps;
	std::map<unsigned, std::vector<TxnEpochStates> > uid_to_rwlogstats;
	void do_insertTxUIDTimeStamp(unsigned uid, char tag);
	void do_appendTxUIDRWLogEntry(unsigned uid, char rw, unsigned addr_tag);
	void do_appendRWSetSize(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi);

	// 2015-10-17: Measuring ``Dark Silicon Power Savings Opportunity''

	std::map<simd_function_unit*, struct FUIdentifier> simdfu_to_ident;
	std::map<simd_function_unit*, struct FUOccupancyState> last_fu_occupancy_state;
};

class CTAID_TID_Ty; // defined in commit_unit.h

class MyThreadEvent {
public:
	MyThreadEvent () : cycle(0) { }
	unsigned long long cycle;
	virtual const char* getEventName() const = 0;
	virtual ~MyThreadEvent() {}
	virtual void print(FILE* f) { printf("[%s @ %llu]", getEventName(), cycle); }
};

class MyThreadInitEvent : public MyThreadEvent {
public:
	unsigned wid; // WID is bound here
	MyThreadInitEvent(unsigned long long x, unsigned _wid) { cycle = x; wid = _wid; }
	virtual const char* getEventName() const { return "ThreadBorn"; }
};

class MyThreadExitEvent : public MyThreadEvent {
public:
	MyThreadExitEvent(unsigned long long x) { cycle = x; }
	virtual const char* getEventName() const { return "ThreadExit"; }
};

class MyThreadBeginTxnEvent : public MyThreadEvent {
public:
	MyThreadBeginTxnEvent(unsigned long long x) { cycle = x; }
	virtual const char* getEventName() const { return "BeginTxn"; }
};

class MyThreadStartScoreboardCommitEvent : public MyThreadEvent {
public:
	MyThreadStartScoreboardCommitEvent(unsigned long long x) { cycle = x; }
	virtual const char* getEventName() const { return "StartScoreboardCommit"; }
};

class MyThreadDoneScoreboardCommitEvent : public MyThreadEvent {
public:
	bool is_passed;
	MyThreadDoneScoreboardCommitEvent(unsigned long long x, bool _is_passed) {
		cycle = x; is_passed = _is_passed;
	}
	virtual const char* getEventName() const { return "DoneScoreboardCommit"; }
};

class MyThreadDoneCoreSideReadonlyCommitEvent : public MyThreadEvent {
public:
	MyThreadDoneCoreSideReadonlyCommitEvent(unsigned long long x) { cycle = x; }
	virtual const char* getEventName() const { return "DoneCoreSideReadonlyCommit"; }
};

class MyThreadFailPreCommitValidation : public MyThreadEvent {
public:
	std::string reason;
	MyThreadFailPreCommitValidation(unsigned long long x, const char* _reason) {
		cycle = x; reason = _reason;
	}
	virtual const char* getEventName() const { return "FailPreCommitValidation"; }
	void print(FILE* f) {
		printf("[%s @ %llu (%s)]", getEventName(), cycle, reason.c_str());
	}
};

class MyThreadScoreboardHazardEvent : public MyThreadEvent {
public:
	std::string reason;
	unsigned long long until;
	MyThreadScoreboardHazardEvent(unsigned long long x, const char* _reason) {
		until = cycle = x; reason = _reason;
	}
	virtual const char* getEventName() const { return "ScoreboardHazard"; }
	void print (FILE* f) {
		fprintf(f, "[%s @ %llu until %llu (%s)]", getEventName(), cycle, until, reason.c_str());
	}
};

class MyThreadActiveMaskChangedEvent : public MyThreadEvent {
public:
	std::string reason;
	char changed_to;
	MyThreadActiveMaskChangedEvent (unsigned long long x, const char* _reason, char _ct) {
		cycle = x; reason = _reason; changed_to = _ct;
	}
	virtual const char* getEventName() const { return "ActiveMaskChanged"; }
	void print (FILE* f) {
		fprintf(f, "[%s @ %llu (changed to %d, reason: %s)]", getEventName(), cycle, changed_to, reason.c_str());
	}
};

class MyThreadTLWStateChangedEvent : public MyThreadEvent {
public:
	/*
	 *    enum commit_tx_state_t {
      IDLE = 0,
      INTRA_WARP_CD,
      ACQ_CID,
      ACQ_CU_ENTRIES,
      WAIT_CU_ALLOC_REPLY,
      SEND_RS,
      SEND_WS,
      WAIT_CU_REPLY,
      RESEND_RS,
      RESEND_WS,
      SEND_ACK_CLEANUP
   };
	 */
	static	const char* ctsz[];
	enum tx_log_walker::commit_tx_state_t state;
	bool is_passed;
	MyThreadTLWStateChangedEvent(unsigned long long x, enum tx_log_walker::commit_tx_state_t _state) {
		cycle = x; state = _state;
		is_passed = false;
	}
	virtual const char* getEventName() const { return "TLWStateChange"; }
	void print (FILE* f) {
		if (state == tx_log_walker::SEND_ACK_CLEANUP) {
			fprintf(f, "[%s @ %llu (%s %s)]",
				getEventName(), cycle, ctsz[(int)state], is_passed ? "PASS" : "FAIL");
		} else {
			fprintf(f, "[%s @ %llu (%s)]", getEventName(), cycle, ctsz[(int)state]);
		}
	}
};


class MyThreadEventParser {
public:
	enum ThdStateParseState {
		EXITED,
		RUNNING,
		IN_SCOREBOARD_COMMIT,
		IN_TLW_CU_COMMUNICATION,
	};
	void add(const MyThreadEventParser& other) {
		num_init += other.num_init;
		num_scoreboard_commit += other.num_scoreboard_commit;
		num_pre_cu_abort += other.num_pre_cu_abort;
		cycles += other.cycles; cycles_inactive += other.cycles_inactive;
		num_tlw_state_change += other.num_tlw_state_change;
		num_tx += other.num_tx;
		num_cu_aborts += other.num_cu_aborts;
		num_commits += other.num_commits;
		num_active_mask_change += other.num_active_mask_change;
		commit_cycle_pass += other.commit_cycle_pass;
		commit_cycle_fail += other.commit_cycle_fail;
		inactive_cycle_incommit += other.inactive_cycle_incommit;
		for (std::map<std::string, long>::const_iterator itr = other.cycle_hazards.begin();
			itr != other.cycle_hazards.end(); itr++) {
			if (cycle_hazards.find(itr->first) == cycle_hazards.end())
				cycle_hazards[itr->first] = 0;
			cycle_hazards[itr->first] += itr->second;
		}
	}

	MyThreadEventParser() {
		num_init = 0, num_scoreboard_commit = 0, num_pre_cu_abort = 0, cycles = 0,
		num_tlw_state_change = 0, num_tx = 0, num_cu_aborts = 0, num_commits = 0,
		num_active_mask_change = 0, cycles_inactive = 0;
		is_parse_error = 0;
		commit_cycle_pass = commit_cycle_fail = 0;
		inactive_cycle_incommit = 0;

		// Parsing state machine
		is_active = true;
		state = EXITED;
		prev_event = curr_event = NULL;
		curr_tx_pass = -999; prev_exec_begin = -999; prev_commit_begin = -999;
		commit_tx_state = tx_log_walker::IDLE;

		prev_donecommit_inactive = -999;
	}
	int is_parse_error, curr_tx_pass;
	unsigned num_init, num_scoreboard_commit, num_pre_cu_abort, num_tlw_state_change,
		num_tx, num_cu_aborts, num_commits, num_active_mask_change;
	long cycles, cycles_inactive;
	long inactive_cycle_incommit;

	/*
	 *  |-Active
	 *  |   |---- Scoreboard Hazard
	 *  |   |                   |----Pass
	 *  |   |                   |----Fail
	 *  |   |                   |----Non Txn
	 *  |   |---- Committing
	 *  |   |            |----Pass
	 *  |   |            |----Fail
	 *  |   |---- Execution
	 *  |                |----Pass
	 *  |                |----Fail
	 *  |                |----Non Txn
	 *  |-Inactive
	 *      |--- Warp Divergence
	 *      |--- Waiting for other warps to complete
	 */

	// TLW-states
	long commit_cycle_pass, commit_cycle_fail;
	std::map<std::string, long> cycle_hazards;


	// States used in parsing
	long prev_exec_begin, prev_commit_begin, prev_donecommit_inactive;
	bool is_active;
	tx_log_walker::commit_tx_state_t commit_tx_state;
	ThdStateParseState state;
	MyThreadEvent* prev_event, *curr_event;

	void singleStepEvent(MyThreadEvent* evt);
	MyThreadEvent* getCurrentEvent() { return curr_event; }
	void do_advanceEvent(MyThreadEvent* evt);
	void handleCurrEvent(MyThreadEvent* currevt);
	void finalize();
	void printHeader(FILE* f);
	void print(FILE* f, double mult);
	long getTotalHazardCycles();
	unsigned long computeExecCycle() { return cycles - cycles_inactive
		- commit_cycle_pass - commit_cycle_fail - getTotalHazardCycles(); }
	unsigned long computeCommitCycle() {
		return commit_cycle_pass + commit_cycle_fail; }
};


// 2015-07-11: Enumerate all occurrences of L1 accesses.
class CartographerTimeSeries {
friend class gpgpu_sim;
friend class Cartographer;
friend class GreatCartographer;
public:
	CartographerTimeSeries() {
		struct MyL1Stats s = {};
		this->l1_stats = s;
		verified = true;
		interval = 100000;
		curr_l1_interval_begin = 0; // Sentry for L1
		curr_insts_interval_begin = 0; // Sentry for Insn throughput
		num_insts_completed = 0;
		is_always_print_all_events = log_events = account_every_cycle =
			account_committing_warps = dump_txn_rwsets = false;
		f_rwsets = NULL;
		max_cid_in_flight = num_linked_with_sandwich = 0;
	}

	struct TxnEventEntry {
		MyTxnEvent event;
		dim3 ctaid, tid;
		unsigned num_transaction;
		unsigned long long cycle;
	};
	std::list<TxnEventEntry> events;
	void onPtxThreadStartTransaction(ptx_thread_info* thread);
	void onPtxThreadRollbackTransaction(ptx_thread_info* thread);
	void onPtxThreadCommitTransaction(ptx_thread_info* thread);

	void incrementL1AccessCount(const char* which, int status);
	void summary();
	bool account_every_cycle, verified, is_always_print_all_events, log_events,
		account_committing_warps, dump_txn_rwsets;

	char parsing_mode = 1; // 1: Store into list then parse at the end;
	                        // 2: Parse in-place, do not store events

	struct MyL1Stats {
		unsigned long
			l1_access_count,   l1_rf_count,
			l1c_access_count,  l1c_rf_count,
			l1i_access_count,  l1i_rf_count,
			l1t_access_count,  l1t_rf_count,
			l1d_access_count,  l1d_rf_count,
			l1d_access_count_txn, l1d_rf_count_txn;
	};
	struct MyL1Stats l1_stats;
	std::list<struct MyL1Stats> l1_per_interval;
	unsigned interval; // in cycles
	void checkL1Accesses(unsigned l1_tot, unsigned l1c, unsigned l1i, unsigned l1t, unsigned l1d);
	void checkL1ReservationFails(unsigned l1_rf, unsigned l1c_rf, unsigned l1i_rf, unsigned l1t_rf, unsigned l1d_rf);

	// Instruction Throughput
	unsigned long num_insts_completed;
	std::list<unsigned long> num_insts_per_interval;
	void incrementInstCount(unsigned count);

	void initRWListFile(const char* filename) {
		f_rwsets = gzopen(filename, "wb");
		fn_rwsets = filename;
		if (!f_rwsets) {
			assert(0 && "Txn R/W List File Open ERROR !!!!!!");
		}
	}

	void closeRWListFile() {
		if (f_rwsets) {
			const unsigned long long fin = (unsigned long long)(-1);
			gzwrite(f_rwsets, &fin, sizeof(unsigned long long));
			gzclose(f_rwsets);
			printf("[CartographerTimeSeries] dumped Read/Write set to file %s\n",
				fn_rwsets.c_str());
		}
	}

	unsigned max_cid_in_flight;
	unsigned num_linked_with_sandwich;
	std::vector<MyCommittingTxnEntry*> txn_cid_in_flight, txn_cid_done;
	gzFile f_rwsets;
	std::map<unsigned, addr_set_t> read_sets, write_sets;
	void printAllReadSets();
	void printReadSet(unsigned cid);
	std::string fn_rwsets;

	// itr->core->m_sid == sid &&
	// itr->wid == wid &&
	// itr->cid == cid
	MyCommittingTxnEntry* locateMyCommittingTxnEntryByCID(unsigned commit_id);

	// At this call = ACQ_CID ---> SEND_RS
	void onTxnAcquiredCID(shader_core_ctx* core, unsigned wid,
		struct tx_log_walker::warp_commit_tx_t* commit_wp_info, struct tx_log_walker::commit_tx_t* tx,
		int tid_in_warp);
	void onTxnStartSendingWS(unsigned commit_id);
	void onTxnSendOneRSEntry(unsigned commit_id);
	void onTxnSendOneWSEntry(unsigned commit_id);
	void onTxnDoneFill(unsigned cid);
	void onTxnReceivedCUReply(unsigned commit_id);
	void onTxnSendTXPassFail(unsigned commit_id);

	// At this call = one RS entry is sent to the CU

	// commit_tx_t* may be reclaimed, so we need to pass in the ptx_thread_info* as well
	void onTxnReceivedAllCUReplies(shader_core_ctx* core, unsigned wid,
		struct tx_log_walker::warp_commit_tx_t* commit_wp_info, struct tx_log_walker::commit_tx_t* tx,
		const ptx_thread_info* thread, int tid_in_warp);
	void getCommittingTxnsInfo(std::vector<shader_core_ctx*>* cores, std::vector<unsigned>* wids,
		std::vector<struct tx_log_walker::warp_commit_tx_t>* commit_warps, bool is_done_fill_only);

	void getCommittingTxnsInfo_2(std::vector<shader_core_ctx*>* cores, std::vector<unsigned>* wids,
			std::vector<struct tx_log_walker::warp_commit_tx_t>* commit_warps, std::vector<unsigned>* cids,
			std::vector<unsigned long long>* atimes,
			bool is_done_fill_only);

	void getCommitTxInfoFromCommitID(unsigned cid,
			tx_log_walker::warp_commit_tx_t** txt, shader_core_ctx** core, unsigned long long* atime);


	// Book-keeping of the transaction epochs that ever existed
	// PER-THREAD !
	struct MyTransactionThreadLifeTimeEntry {
		unsigned pc;
		unsigned epoch; // +1 when rollback
		addr_set_t read_set, write_set;
		bool is_passed;
		ptx_instruction* ptx_inst;
	};
	std::map<CTAID_TID_Ty, std::list<struct MyTransactionThreadLifeTimeEntry*>> m_done_txn_lifetime_entries; // Say "1" b/c data is from the functional simulation side
	std::map<CTAID_TID_Ty, struct MyTransactionThreadLifeTimeEntry*> m_live_txn_lifetime_entries; // Data from the timing simulation side

	struct MyTransactionWarpLifeTimeEntry {
		unsigned char state;
	};

	enum MyThreadState {
		THD_INACTIVE,
		THD_RUNNING,
		THD_TXN_EXEC,
		THD_TXN_COMMITTING,
		NUM_MYTHREADSTATE
	};
	// Similar to this one:
	// void void  onInstructionIssued(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst, warp_inst_t* d_inst, const char* fu); // Modifies the SN of the Dyn Inst!
	void onTxBeginIssuedToFU(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst, const warp_inst_t* to_inst);
	void onTxCommitIssuedToFU(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst, const warp_inst_t* to_inst);
	void accountAllThreads();
	void dumpThreadStateHist();


	// Event List and functions --------------------------------------------------------------------------------------------

	// Some are warp level, some are thread level
	// Some are from func sim side, some are from timing sim side

	// This one is called through issueBlock2Core
	void onPTXInitThread(core_t* core, dim3 ctaid, dim3 tid, unsigned hw_wid);
	void onThreadExit(ptx_thread_info* pti);
	void onStartScoreboardCommit(shader_core_ctx* core, unsigned warp_id, warp_inst_t* inst);
	void onDoneScoreboardCommit(shader_core_ctx* core, unsigned warp_idm, warp_inst_t* inst);
	void onThreadDoneCoreSideCommit(shader_core_ctx* core, ptx_thread_info* pti);
	void onThreadFailedPreCommitValidation(shader_core_ctx* core, ptx_thread_info* pti, const char* reason);
	void onWarpScoreboardHazard(shader_core_ctx* core, unsigned wid, const char* reason);
	void onRemoveAllWarpToTIDMapOnCore (core_t* core);
	void onThreadTLWStateChanged(shader_core_ctx* core, ptx_thread_info* pti, enum tx_log_walker::commit_tx_state_t _state);
	void onThreadSendTxPassFail(shader_core_ctx* core, ptx_thread_info* pti, enum tx_log_walker::commit_tx_state_t _state, bool is_pass);
	void onWarpActiveMaskChanged(core_t* core, unsigned warp_id, const simt_mask_t* mask, const char* reason);

	struct MyThreadStateChangeTy {
		enum MyThreadState state0, state1;
		unsigned long long cycle;
	};

	void onTommyPacketSent(shader_core_ctx* core, unsigned wid, unsigned cid, unsigned tid_in_warp);
	void dumpAll();

private:
	void do_removeInFlightCIDEntry(shader_core_ctx* core, unsigned wid, unsigned cid, unsigned tid_in_warp);

	enum MyThreadState getMyThreadState (ptx_thread_info* ptx_thd);
	void do_incrementL1AccessCount(const char* which, int status, struct MyL1Stats* stats);
	unsigned curr_l1_interval_begin;
	unsigned curr_insts_interval_begin;
	std::deque<std::pair<unsigned long long, std::map<enum MyThreadState, unsigned> > > thd_status_history_ref;
	std::map<CTAID_TID_Ty, std::map<std::string, unsigned> > thd_cycle_breakdown_ref;

	std::unordered_map<CTAID_TID_Ty, std::deque<MyThreadEvent*>> thd_events;
	std::unordered_map<CTAID_TID_Ty, MyThreadEventParser*> thd_event_parsers;

	std::map<CTAID_TID_Ty, enum MyThreadState> prev_thd_state;
	std::map<CTAID_TID_Ty, std::list<struct MyThreadStateChangeTy> > thd_state_change_history;
	std::map<std::pair<unsigned, unsigned>, std::set<CTAID_TID_Ty> > sid_wid_to_threads;

	void appendEvent (const CTAID_TID_Ty&, MyThreadEvent*);
	MyThreadEvent* getCurrEvent (const CTAID_TID_Ty&);

	void do_onTxnSendOneRWEntry(unsigned commit_id, char rw);
};

class CartographerMem {
friend class GreatCartographer;
public:
	CartographerMem() {
		is_dump_all = false;
		sent_count = rcvd_count = 0;
		traffic_breakdown_sent = new traffic_breakdown("CartographerMem[sent]");
		traffic_breakdown_rcvd = new traffic_breakdown("CartographerMem[rcvd]");
		account_every_cycle = false;
		sprintf(mem_hist_db_filename, "mem_hist.db");
		f_mem_fetch_state_transition = NULL;
	}
	void initMemFetchStateTransitionFile() {
		f_mem_fetch_state_transition = fopen(fn_mem_fetch_state_transition.c_str(), "w");
		if (!f_mem_fetch_state_transition) { assert(false && "Error opening mem fetch state transition dot file !\n"); }
	}
	std::deque<MyMemFetchInfo> mem_fetches;
	bool is_dump_all;
	unsigned sent_count, rcvd_count;
	void onMemFetchInCacheMissQueue (shader_core_ctx* shader, cache_t* cache, warp_inst_t* inst, mem_fetch* mf);
	void onMemFetchInTLWMessageQueue (shader_core_ctx* shader, tx_log_walker* tlw,   warp_inst_t* inst, mem_fetch* mf);
	void onMemFetchInLDSTOutbound (shader_core_ctx* shader, ldst_unit* ldst, warp_inst_t* inst, mem_fetch* mf);
	void onMemFetchReceivedFromIcnt (mem_fetch* mf);
	void summary();
	void accountAllMemFetches();
	void dumpMemFetchHist();
	bool account_every_cycle;
	char mem_hist_db_filename[100];

	// Mem Fetch State Transition
	bool is_log_mem_fetch_state_transition;
	std::string fn_mem_fetch_state_transition;
	FILE* f_mem_fetch_state_transition;

private:
	std::list<std::pair<unsigned long long, std::map<enum mem_fetch_status, unsigned> > > mf_status_hist; // histogram!
	std::map<enum mem_fetch_status, unsigned long> mf_state_histo_cumsum;

	// Mem fetch state transition datastructures.
	std::unordered_map<mem_fetch*, enum mem_fetch_status> mf_prev_state;
	std::map<std::pair<enum mem_fetch_status, enum mem_fetch_status>, unsigned> mf_state_transition_count;

	traffic_breakdown* traffic_breakdown_sent, *traffic_breakdown_rcvd;
	void do_onNewMemFetch(shader_core_ctx* shader, warp_inst_t* inst, mem_fetch* mf, bool is_l2);
};

#endif

#include "Cartographer.h"
#include <dlfcn.h>
#include <sys/time.h>

// 08-12: Had this problem on MUM:
//
// The first instruction has not issued. Y?
//
// [!FindWarpInstAtPPLStageEx multiple results! (2,4) inst=0xe475b0 stage=1
// Warp [2,4]
// [ PC=0xa18 (_1.ptx:630) ld.param.u64 %rl9, [_Z17mummergpuRCKernelP10MatchCoordPcPKiS3_ii_param_0];](0xe475b0) PC=A18 FB=307573 Fd=1 Sch=0 SchC=307574 Ex=0 Ret=0 FU=(null) StallFU=8672912 Note=0 St=1 SN=6
// [ PC=0x9f0 (_1.ptx:602) mov.u32 %r138, %r40;](0x884350) PC=9F0 FB=619019 Fd=1 Sch=619021 SchC=619020 Ex=619026 Ret=0 FU=SP StallFU=619021 Note=0 St=99 SN=3
// [ PC=0x9f8 (_1.ptx:603) mov.u16 %rs39, %rs17;](0x859af0) PC=9F8 FB=619019 Fd=1 Sch=619022 SchC=619020 Ex=619026 Ret=0 FU=SP StallFU=619022 Note=0 St=99 SN=4
// [ PC=0xa00 (_1.ptx:604) mov.u16 %rs56, %rs18;](0x855cf0) PC=A00 FB=619022 Fd=1 Sch=619024 SchC=619023 Ex=0 Ret=0 FU=SP StallFU=619024 Note=0 St=2 SN=1
// [ PC=0xa08 (_1.ptx:606) @%p20 bra BB3_3;](0x857790) PC=A08 FB=619022 Fd=1 Sch=619025 SchC=619023 Ex=0 Ret=0 FU=SP StallFU=619025 Note=0 St=2 SN=5
// [ PC=0xa18 (_1.ptx:630) ld.param.u64 %rl9, [_Z17mummergpuRCKernelP10MatchCoordPcPKiS3_ii_param_0];](0xe475b0) PC=A18 FB=619025 Fd=1 Sch=0 SchC=619026 Ex=0 Ret=0 FU=(null) StallFU=8672912 Note=0 St=1 SN=7
//
// This morning's code changes are __not correct__ b/c AES crashes after 6.2 million instructions.
//

extern gpgpu_sim *g_the_gpu;
bool is_running_as_tool = false; // if is running as a tool then dtor and ctor aren't executed.
bool is_disable_aug19 = true;
bool is_disable_ldstdelay = true;
bool is_log_inst_mix = true;

static const int NUM_SHADERS = 41;
static const int NUM_WARPS_PER_SHADER = 48;

bool operator==(struct MaskState& lhs, struct MaskState& rhs) {
	return (lhs.the_mask == rhs.the_mask);
}

#include "gpgpu-sim/shader.h"

#define MF_TUP_BEGIN(X) static const char* status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) };
#include "gpgpu-sim/mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

long ilti_serial_id = 0;
const char* Reasons[] = {
	"Finished normally",
	"Flushed due to control hazard",
};

unsigned long *instr_mix; // overall
unsigned long g_num_plus1_schedule, g_num_plus1_ldst; // Aug 30: How many +1 cycle delay is added to Scheduler?

class shader_core_ctx;
struct ifetch_buffer_t;

const int ENABLE_CARTOGRAPHER = 0;

unsigned long long g_scoreboard_wp_acc_orig; // Number of accesses to SIMT stack information in "original" mode
unsigned long long g_scoreboard_wp_acc_pred; // Number of accesses to SIMT stack information in "predicated" mode
unsigned long long g_scoreboard_wp_acc_pred1; // Pred plus function unit constraints

unsigned long long g_coalesce_acc_orig; // Simply the number of "thread*address"
unsigned long long g_coalesce_acc_newcomers; // Newcomer. Is the result of "scanning" through thd 0 through 31. In other words, # times a std::set is inserted into.

Cartographer*    g_cartographer;
CartographerTimeSeries* g_cartographerts;
CartographerMem* g_cartographermem;

int g_ldst_depth = 3;
static bool is_log_coalesced_addrs = false;
static unsigned long coalesced_db_write_interval = 10000; // Write DB every 100000 entries.

static bool g_dump_ring_ids = false;
FILE* f_ring_ids;

static char* warp_info_thin_log_filename = NULL;
static char* simtstack_log_filename = NULL;
static char* warp_scheduled_filename = NULL;
static char* sandwich_plot_filename = NULL;
static char* sandwich_db_filename = NULL; // has to end with ".db"

static unsigned char PPL_AFTER_FETCH = 0;
static unsigned char PPL_AFTER_DECODE = 1;
static unsigned char PPL_AFTER_ISSUE = 2;
static unsigned char PPL_AFTER_ISSUE_TO_FU = 3;
static unsigned char PPL_LDST_READY = 4;

static unsigned long long getCurrentCycleCount() {
	return gpu_tot_sim_cycle + gpu_sim_cycle;
}

std::map<unsigned, std::string> g_pc_to_ptxsource;
bool Cartographer_lookupPTXSource(unsigned pc, std::string* sz) {
	if (g_pc_to_ptxsource.find(pc) == g_pc_to_ptxsource.end()) {
		return false;
	} else {
		*sz = g_pc_to_ptxsource.at(pc);
		return true;
	}
}

// Preparation for the oracle txn sched ...
std::set<unsigned> g_txn_cid_in_flight;
unsigned tommy_0729_delay = 0, tommy_0729_delay1 = 233; // delay1 considers the time needed for the commit to propagate into GMEM
unsigned tommy_0729_global_capacity = 1000000; // Number of addresses, globally
unsigned tommy_cat_acc_addremove = 0; // Number of access to the CAT (in Shader Cores) for adding/removing entries;
unsigned tommy_cat_acc_lookup    = 0; // Number of access to the CAT (in Shader Cores) for looking up an entry;
unsigned tommy_rct_acc           = 0; // Number of access to the Reference Count Table (in Commit Units)
extern unsigned num_tommy_packets, cum_tommy_packet_delay;
extern void PrintAllHistoryAddrOwnerInfo();

// 2015-08-09
unsigned g_stagger1_count = 0;  // # of calls to __staggertxns().
// 2015-08-10
unsigned g_stagger2_count = 0; // # of staggers of transactional loads and stores.

unsigned g_num_staggers = 0, g_num_stagger_aborts = 0,
		g_num_stagger_samewarp = 0, g_watched_tid = (unsigned)(-1);

int g_print_simt_stack_sid = -999, g_print_simt_stack_wid = -999;
unsigned g_iwcd_shmem_count = 0, g_iwcd_rwlogaccess_count = 0;
int g_tommy_reconverge_point_mode = 1;
int g_tommy_break_cycle = INT_MAX;
bool g_tommy_flag_0808 = false, g_tommy_flag_0729 = false, g_tommy_flag_0808_1 = false;
int g_tommy_flag_1028 = 0;
unsigned long g_cat_look_delays = 0, g_cat_look_hazards = 0;
unsigned long g_intrawarpcd_l1access = 0;
bool g_tommy_dbg_0830 = false;
unsigned g_0830_sanity_check_count = 0, g_0830_n_count = 0, g_0830_t_count = 0;
unsigned g_0830_restore_by_pc_count = 0;

// Last round of data-gathering before submitting to hpca16
bool g_tommy_log_0902 = true;

// Adding modelling of interconnect delays
bool g_tommy_flag_1124 = false;
unsigned g_tommy_packet_count = 0, g_tommy_packet_count_tout = 0; // second count is for timeout-induced
unsigned long long g_last_tommy_packet_ts = 0;

#define DBG(stmt) ;
// -------------------------------------------------------------------
// Logging Functions
// --------------------------------------------------------------------


static std::vector<std::string> split(const std::string& s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}


// ----------------------
std::string InstLifeTimeInfo::serializeScoreboardDeps() {
	int idx = 0;
	std::string ret;
	std::set<long>::iterator itr;
	for(itr = scoreboard_deps.begin(); itr != scoreboard_deps.end(); itr++) {
		long k = *itr;
		char c[20];
		sprintf(c, "%ld", k);
		if(idx > 0) ret = ret + ":";
		ret = ret + c;
		idx++;
	}
	return ret;
}
void InstLifeTimeInfo::deserializeScoreboardDeps(std::string str) {
	std::vector<std::string> sbdeps;
	sbdeps = split(str, ',', sbdeps);
	for(std::vector<std::string>::iterator itr = sbdeps.begin();
			itr != sbdeps.end(); itr++) {
		std::string sz = *itr;
		long k = atol(sz.c_str());
		scoreboard_deps.insert(k);
	}
}
std::string InstLifeTimeInfo::serializeFUStalledBys() {
	std::string ret;
	std::string tmp;

	#define FOO_0(initvalue, fromwhich) \
	{ \
		tmp = initvalue; \
		int idx = 0; \
		std::set<long>::iterator itr; \
		for(itr = fromwhich.begin(); itr != fromwhich.end(); itr++) { \
			long k = *itr;\
			char c[30]; \
			sprintf(c, "%ld", k); \
			if(idx > 0) tmp = tmp + ":"; \
			tmp = tmp + c; \
			idx++; \
		} \
	}

	FOO_0("", mem_stalledby);
	ret = tmp + ";";
	FOO_0("",  sp_stalledby);
	ret += tmp + ";";
	FOO_0("", sfu_stalledby);
	ret += tmp;
	return ret;
}

void InstLifeTimeInfo::deserializeFUStalledBys(std::string str) {
	std::vector<std::string> deps;
	deps = split(str, ';', deps);

	#define FOO_1(from, to) \
	{ \
		if(from.length() >= 1) { \
			std::vector<std::string> dep; \
			dep = split(from, ',', dep); \
			std::vector<std::string>::iterator itr;\
			for(itr = dep.begin(); itr != dep.end(); itr++) { \
				std::string s = *itr; \
				to.insert(atoi(s.c_str())); \
			}\
		} \
	}

	if(deps.size() > 0) {
		FOO_1(deps[0], mem_stalledby);
	}
	if(deps.size() > 1) {
		FOO_1(deps[1], sp_stalledby);
	}
	if(deps.size() > 2) {
		FOO_1(deps[2], sfu_stalledby);
	}
}

// -------------------------

// On set kernel

void Cartographer_onIssueBlockToCore(shader_core_ctx* from,
		unsigned cta_hw_id, unsigned warp_id_min, unsigned warp_id_max, kernel_info_t& kernel) {
	g_cartographer->onIssueBlockToCore(from, cta_hw_id, warp_id_min, warp_id_max, kernel);
}

void
Cartographer::onIssueBlockToCore(shader_core_ctx* from, unsigned cta_hw_id,
		unsigned warp_id_min, unsigned warp_id_max,
		kernel_info_t &kernel) {

}

void Cartographer_onWarpScheduled(shader_core_ctx* shader, unsigned warp_id, char fu) {
	g_cartographer->onWarpScheduled(shader, warp_id, fu);
}
void Cartographer::onWarpScheduled(shader_core_ctx* shader, unsigned warp_id, char fu) {
	unsigned shader_id = shader->get_sid();
	if(!shouldLog(shader_id, warp_id)) return;
	SHADER_WARP_ID_Ty swid = std::make_pair(shader_id, warp_id);
	if(m_warp_scheduled.find(swid) == m_warp_scheduled.end()) {
		WARP_SCHEDULED_TIME_ENTRY_LIST_P_Ty the_new_schlist = new std::list<WARP_SCHEDULED_TIME_ENTRY>();
		m_warp_scheduled[swid] = the_new_schlist;
	}
	WARP_SCHEDULED_TIME_ENTRY ety;
	ety.first = gpu_tot_sim_cycle + gpu_sim_cycle;
	ety.second = fu;
	m_warp_scheduled[swid]->push_back(ety);
}

// --------------------------------------------
// Constructor and Destructor
// --------------------------------------------

__attribute__((constructor))
void TommyEnvConfig() {
	if(is_running_as_tool) return;

	// Cartographer for warp instructions

	// Correction: no need to get data from the function simulator.
	// Need to specify 1) a core. 2) idx of CTA on this core
	//   to log what we want!
	char* is_log = getenv("IS_LOG_BARRIER");
	g_cartographer = new Cartographer();
	g_cartographer->is_log_barrier = false;
	if(is_log) {
		if(strcmp(is_log, "1")==0){
			g_cartographer->is_log_barrier = true;
		}
	}

	if(g_cartographer->is_log_barrier == true) {
		printf("[Cartographer] Write barrier log! (%s)\n", is_log);
	} else {
		printf("[Cartographer] Not writing barrier log!\n");
	}

	char* log_barrierid = getenv("LOG_BARRIER_COREID");
	if(log_barrierid) {
		int coreid = atoi(log_barrierid);
		g_cartographer->log_barrier_coreid = coreid;
		printf("[Cartographer] Writing logs for core %3d\n", coreid);
	} else {
		g_cartographer->log_barrier_coreid = -1;
		printf("[Cartographer] Writing logs for all cores\n");
	}

	char* logfilename = getenv("LOG_BARRIER_FILENAME");
	if(logfilename) {
		g_cartographer->logfile = fopen(logfilename, "w");
		printf("[Cartographer] Writing logs to %s\n", logfilename);
	} else {
		g_cartographer->logfile = stdout;
		printf("[Cartographer] Writing logs to stdout\n");
	}

	// This warp history filename does not take into account kernel launch information!
	warp_info_thin_log_filename = getenv("WARP_HISTORY_FILENAME");
	if(warp_info_thin_log_filename == NULL) {
		printf("[Cartographer] Not exporting warp history info.\n");
	} else {
		printf("[Cartographer] Exporting warp history info to %s\n",
			warp_info_thin_log_filename);
	}

	// June 03
	simtstack_log_filename = getenv("SIMTSTACK_HISTORY_FILENAME");
	if(simtstack_log_filename == NULL) {
		g_cartographer->is_log_simt_states = false;
		printf("[Cartographer] Not exporting simt stack history info.\n");
	} else {
		g_cartographer->is_log_simt_states = true;
		printf("[Cartographer] Exporting simt stack history info to %s (ptr=%p)\n",
			simtstack_log_filename,
			&(g_cartographer->is_log_simt_states));
	}

	warp_scheduled_filename = getenv("WARP_SCHEDULED_FILENAME");
	if(warp_scheduled_filename == NULL) {
		g_cartographer->is_log_warp_sched_evts = false;
		printf("[Cartographer] Not exporting warp being scheduled info.\n");
	} else {
		g_cartographer->is_log_warp_sched_evts = true;
		printf("[Cartographer] Exporting warp being scheduled info to %s\n",
			warp_scheduled_filename);
	}

	// June 05
	sandwich_plot_filename = getenv("SANDWICH_PLOT_FILENAME");
	if(sandwich_plot_filename == NULL) {
		printf("[Cartographer] Not exporting sandwich plot.\n");
	}

	sandwich_db_filename = getenv("SANDWICH_DB_FILENAME");
	if(sandwich_db_filename == NULL) {
		printf("[Cartographer] Not exporting sandwich DB\n");
	}

	{
		char* log_shaders = getenv("LOG_SHADERS");
		if(log_shaders) {
			std::string lss; lss = log_shaders;
			std::vector<std::string> ss = split(lss, ',');
			if(ss.size() > 0) {
				g_cartographer->is_using_whitelist = true;
				printf("[Cartographer] Shader ID whitelist turned on.\n");
				std::vector<std::string>::iterator itr;
				for(itr = ss.begin(); itr != ss.end(); itr++) {
					std::string k = *itr;
					int shaderid = atoi(k.c_str());
					printf("   Logging shader ID %d\n", shaderid);
					g_cartographer->shaders_logged.insert(shaderid);
				}
			} else {
				printf("[Cartographer] Shader ID whitelist turned off (logging all shaders)\n");
			}
		}
		printf("[Cartographer] Shader ID whitelist not defined (logging all shaders)\n");
	}

	{
		char* log_cyc_ints = getenv("LOG_CYCLE_INTERVALS");
		if(log_cyc_ints) {
			std::string lci; lci = log_cyc_ints;
			std::vector<std::string> itvs = split(lci, ':');
			if(itvs.size() > 0) {
				unsigned last_upper = 0;
				for(std::vector<std::string>::iterator itr = itvs.begin();
						itr != itvs.end(); itr++) {
					const std::string& itv = *itr;
					unsigned long long lower=0, upper=0;
					if(sscanf(itv.c_str(), "%llu,%llu", &lower, &upper)==2) {
						if(!((upper >= lower) && (lower >= last_upper))) {
							fprintf(stderr, "Invalid interval: %llu, %llu\n", lower, upper);
							assert(0 && "Invalid interval.\n");
						}
						last_upper = upper;
						printf("   Logging interval [%llu, %llu]\n", lower, upper);
						g_cartographer->cycle_segments_logged.push_back(std::make_pair(lower, upper));
					}
				}
				g_cartographer->curr_whitelist_cycle_seg_itr = g_cartographer->cycle_segments_logged.begin();
			} else {
				printf("[Cartographer] Cycle segments whitelist turned off (log all cycles)\n");
			}
		} else {
			printf("[Cartographer] Cycle segments whitelist not defined (log all cycles)\n");
		}
	}

	{
		char* ldst_depth = getenv("LDST_PIPELINE_DEPTH");
		if(ldst_depth) {
			g_ldst_depth = atoi(ldst_depth);
			printf("[Cartographer] LD/ST pipeline depth set to %d\n", g_ldst_depth);
		}
	}

	{
		char* is_log_coal = getenv("IS_LOG_COALESCED_ADDRS");
		if(is_log_coal) {
			is_log_coalesced_addrs = atoi(is_log_coal);
			printf("[Cartographer] Log coalesced addresses: %d\n", is_log_coalesced_addrs);
			if(is_log_coalesced_addrs) {
				g_cartographer->openAndInitializeCoalescedDB();
			}
		}
	}

	{
		char* is_extra_sch_delay = getenv("IS_EXTRA_CYCLE_SCHEDULER_DELAY");
		if(is_extra_sch_delay) {
			is_disable_aug19 = !(atoi(is_extra_sch_delay));
		}
		printf("[Cartographer] Is extra cycle Scheduler delay: %d\n", !is_disable_aug19);
	}

	{
		char* is_extra_ldst_delay = getenv("IS_EXTRA_CYCLE_LDST_DELAY");
		if(is_extra_ldst_delay) {
			is_disable_ldstdelay = !(atoi(is_extra_ldst_delay));
		}
		printf("[Cartographer] Is extra cycle LDST delay: %d\n", !is_disable_aug19);
	}

	instr_mix = (unsigned long*)(malloc(sizeof(unsigned long)*200));
	for(int i=0; i<200; i++) instr_mix[i] = 0;
	g_num_plus1_schedule = 0;
	g_num_plus1_ldst     = 0;

	// ---------------- Cartographer for Transactions
	g_cartographerts = new CartographerTimeSeries();
	g_cartographer->linked_cartographerts = g_cartographerts;
	{
		char* x = getenv("CARTOGRAPHERTS_EVERYCYCLE");
		if (x) {
			if (atoi(x)==1) g_cartographerts->account_every_cycle = true;
			else g_cartographerts->account_every_cycle = false;
		}

		x = getenv("CARTOGRAPHERTS_PRINTALLEVENTS");
		if (x) { g_cartographerts->is_always_print_all_events = true; }

		x = getenv("CARTOGRAPHERTS_LOGEVENTS");
		if (x) {
			if(atoi(x)==1) g_cartographerts->log_events = true;
			else g_cartographerts->log_events = false;
		}

		x = getenv("CARTOGRAPHERTS_PARSING_MODE");
		if (x) {
			if (atoi(x) == 1) g_cartographerts->parsing_mode = 1;
			else if (atoi(x) == 2) g_cartographerts->parsing_mode = 2;
			else assert(0);
		}

		x = getenv("ACCOUNT_COMMITTING_WARPS");
		if (x) { g_cartographerts->account_committing_warps = true; }
		x = getenv("TOMMY_FLAG_0713");
		if (x) { g_cartographerts->account_committing_warps = true; }
		x = getenv("TOMMY_FLAG_0723");
		if (x) { g_cartographerts->account_committing_warps = true; }

		x = getenv("TM_RWSETS_FILENAME");
		if (x) {
			g_cartographerts->account_committing_warps = true;
			g_cartographerts->dump_txn_rwsets = true;
			g_cartographerts->initRWListFile(x);
		}
	}

	{
		char* x = getenv("TS_INTERVAL");
		if(x) {
			g_cartographerts->interval = atoi(x);
		}
		printf("[CartographerTimeSeries] Interval set to: %u\n", g_cartographerts->interval);
		assert (g_cartographerts->interval > 0);
	}

	// ---------------- Cartographer for Memory requests
	{
		g_cartographermem = new CartographerMem();
		char* x = getenv("CARTOGRAPHERMEM_DUMPALL");
		if (x) {
			g_cartographermem->is_dump_all = true;
		}
	}

	{
		char* x = getenv("ENABLE_CARTOGRAPHERMEM");
		if (x) { if (atoi(x) == 1) g_cartographermem->account_every_cycle = true; }

		x = getenv("MF_STATE_TRANSITION_FILENAME");
		if (x) {
			g_cartographermem->account_every_cycle = true;
			g_cartographermem->is_log_mem_fetch_state_transition = true;
			g_cartographermem->fn_mem_fetch_state_transition = x;
			g_cartographermem->initMemFetchStateTransitionFile();
		}
	}

	{
		char* x = getenv("TOMMY_DUMP_RING_IDS");
		if (x) {
			g_dump_ring_ids = true;
			f_ring_ids = fopen("ring_ids.txt", "w");
			assert(f_ring_ids);
		}
	}

	{
		char* x = getenv("TOMMY_FLAG_0729_DELAY");
		if (x) { int blah = atoi(x); assert(blah >= 0); tommy_0729_delay = blah; }
		x = getenv("TOMMY_FLAG_0729_DELAY1");
		if (x) { int blah = atoi(x); assert(blah >= 0); tommy_0729_delay1 = blah; }
		x = getenv("TOMMY_FLAG_0729_CAPACITY");
		if (x) {
			int blah = atoi(x); assert(blah >= 0); tommy_0729_global_capacity = blah;
			printf("TABLE_0729 size limit: %u entries\n", tommy_0729_global_capacity);
		}
	}

	{
		char* x = getenv("STAGGER1_COUNT");
		if (x) { g_stagger1_count = atoi(x); }
		x = getenv("STAGGER2_COUNT");
		if (x) { g_stagger2_count = atoi(x); }
	}

	{
		char* x = getenv("WATCHED_TID");
		if(x) g_watched_tid = atoi(x);
		x = getenv("TOMMY_PRINT_SIMT_STACK");
		if(x) {
			if (sscanf(x, "%d,%d", &g_print_simt_stack_sid, &g_print_simt_stack_wid) == 2) {
				printf("Will print the SIMT Stack of S%dW%d\n", g_print_simt_stack_sid, g_print_simt_stack_wid);
			}
		}
	}

	{
		char* x = getenv("TOMMY_RECONVERGE_POINT_MODE");
		if (x) { g_tommy_reconverge_point_mode = atoi(x); }
	}
	{
		char* x = getenv("BREAK_CYCLE");
		if (x) { g_tommy_break_cycle = atoi(x); }
	}
	{
		char* x = getenv("TOMMY_FLAG_0808");
		if (x) { g_tommy_flag_0808 = true; }
		x = getenv("TOMMY_FLAG_0808_1");
		if (x) { g_tommy_flag_0808_1 = true; }
		x = getenv("TOMMY_FLAG_0729");
		if (x) { g_tommy_flag_0729 = true; }
		x = getenv("TOMMY_FLAG_1028");
		if (x) { g_tommy_flag_1028 = atoi(x); }
		x = getenv("TOMMY_FLAG_1124");
		if (x) { g_tommy_flag_1124 = true; }
	}
	{
		char* x = getenv("TOMMY_DBG_0830");
		if (x) { g_tommy_dbg_0830 = true; }
		x = getenv("TOMMY_LOG_0902");
		if (x) { g_tommy_log_0902 = (bool)(atoi(x)); }
	}
	{
		// Hack: change tm_warp_info::write_log_offset
		char* x = getenv("WRITE_LOG_OFFSET");
		if (x) { tm_warp_info::write_log_offset = atoi(x); }
	}
	{ // 2015-10-17
		char* x = getenv("TOMMY_FLAG_1017");
		if (x) { g_cartographer->is_log_simdfu_occupancy = true; }
	}
}

void Cartographer_DumpRingIDs(int at_head_old, int at_head_new,// the youngest commit id in the unit
		int fcd_old, int fcd_new, // the oldest commit id that has yet to pass fast conflict detection
		int pass_old, int pass_new, // the oldest commit id that has yet to validate or pass
		int retire_old, int retire_new, // the oldest commit id that has retired
		int commit_old, int commit_new,  // the oldest commit id that has yet to send writeset for committing) {
		int partition_id) {
	if (!g_dump_ring_ids) return;
	fprintf(f_ring_ids, "%llu, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
		getCurrentCycleCount(), partition_id, at_head_old, at_head_new,
		fcd_old, fcd_new, pass_old, pass_new, retire_old, retire_new, commit_old, commit_new);
}

__attribute__((destructor))
void TommyDestructor() {
	if(is_running_as_tool) return;
	printf("[TommyDestructor] destructor called.\n");
	if(g_cartographer->logfile != stdout) {
		fflush(g_cartographer->logfile);
		fclose(g_cartographer->logfile);
	}

	if(warp_info_thin_log_filename!=NULL && strlen(warp_info_thin_log_filename)>0)
		g_cartographer->writeWarpsOnOffHistory(warp_info_thin_log_filename);
	if(simtstack_log_filename != NULL && strlen(simtstack_log_filename) > 0)
		g_cartographer->writeSimtStackHistory(simtstack_log_filename);
	if(warp_scheduled_filename != NULL && strlen(warp_scheduled_filename) > 0)
		g_cartographer->writeWarpScheduledHistory(warp_scheduled_filename);


	printf("[Accesses to warp stats by scheduler] %llu / %llu / %llu\n",
			g_scoreboard_wp_acc_orig, g_scoreboard_wp_acc_pred, g_scoreboard_wp_acc_pred1);
	printf("[Insertions into block address C.A.M.] %llu / %llu\n",
			g_coalesce_acc_orig, g_coalesce_acc_newcomers);
	printf("[Delays added to scheduler] %lu\n", g_num_plus1_schedule);
	printf("[Delays added to ldst] %lu\n", g_num_plus1_ldst);

	g_cartographer->summary();
	g_cartographerts->summary();
	g_cartographermem->summary();

	PrintAllHistoryAddrOwnerInfo();

	if (g_cartographerts->dump_txn_rwsets) {
		g_cartographerts->closeRWListFile();
	}

	if (g_dump_ring_ids == true) {
		fclose(f_ring_ids);
	}

	{
		printf("# of TommyPackets = %u, cum. delay = %u, avg delay = %.2f\n",
			num_tommy_packets, cum_tommy_packet_delay, cum_tommy_packet_delay*1.0/num_tommy_packets);
		printf("# of aborted/samewarp/all staggers = %u / %u / %u\n",
			g_num_stagger_aborts, g_num_stagger_samewarp, g_num_staggers);
		if (g_tommy_dbg_0830) {
			printf("# of Sanity Check 0830's: %u\n", g_0830_sanity_check_count);
			printf("Avg # of N-entrys on top of the topmost T entry: %f = %u / %u\n",
					1.0 * g_0830_n_count / g_0830_sanity_check_count, g_0830_n_count, g_0830_sanity_check_count);
			printf("Avg # of T-entrys: %f = %u / %u\n",
					(1.0 * g_0830_t_count) / g_0830_sanity_check_count, g_0830_t_count, g_0830_sanity_check_count);
		}
		printf("# of restoreAll by PC: %u\n", g_0830_restore_by_pc_count);
		printf("# of 32-cycle delays caused by CAT lookup at LD/ST: %ld\n", g_cat_look_delays);
		printf("# of hazards caused by CAT lookup at LD/ST:         %ld\n", g_cat_look_hazards);
	}
}

void Cartographer::writeWarpsOnOffHistory(char* fn) {
	FILE* file = fopen(fn, "w");
	if(!file) file = stderr;

	fprintf(file, "CoreID\tWarpID\tBegin\tEnd\n");

	printf("[Cartographer] thread-on-off-status statistics!\n");
	for(std::map<unsigned, std::vector<std::vector<std::pair<unsigned long long, unsigned long long> > > >::iterator itr
			= Cartographer::activity_onoff.begin();
			itr!=Cartographer::activity_onoff.end(); itr++) {
		std::vector<std::vector<std::pair<unsigned long long, unsigned long long> > > &core_hist = itr->second;
		unsigned shader_id = itr->first;
		unsigned warp_id = 0;
		for(std::vector<std::vector<std::pair<unsigned long long, unsigned long long> > >::iterator itr2 =
				core_hist.begin(); itr2 != core_hist.end(); itr2++) {
			std::vector<std::pair<unsigned long long, unsigned long long> > &warp_hist = *itr2;
			for(std::vector<std::pair<unsigned long long, unsigned long long> >::iterator itr3 =
					warp_hist.begin(); itr3 != warp_hist.end(); itr3++) {
				unsigned long long begin, end;
				begin = itr3->first; end = itr3->second;
				fprintf(file, "%u\t%u\t%llu\t%llu\n",
						shader_id, warp_id, begin, end);
			}
			warp_id++;
		}
	}

	fflush(file);
	fclose(file);
	printf("[Cartographer] Warp history written to %s\n", fn);
}

void Cartographer::writeSimtStackHistory(char* file_name) {
	printf("[Cartographer &is_log_simt_states = %p\n", &is_log_simt_states);
	assert(is_log_simt_states);
	FILE* file = fopen(file_name, "w");
	if(!file) file = stderr;
	fprintf(file, "CoreID\tWarpID\tMask_ULong\tMask_Count\tBegin\tLength\n");
	MASK_HIST_Ty::iterator itr0 = m_masks.begin();
	MASK_TIME_Ty::iterator itr1 = m_mask_timestamps.begin();
	for(; itr0 != m_masks.end() && itr1 != m_mask_timestamps.end(); itr0++, itr1++) {
		MASK_HIST_ENTRY_P_Ty& rec0 = itr0->second;
		MASK_TIME_ENTRY_P_Ty& rec1 = itr1->second;
		assert(rec0->size() == rec1->size());
		const std::pair<unsigned, unsigned>& swid = itr0->first;
		// For a [warp, core]
		unsigned shader_id = swid.first;
		unsigned warp_id   = swid.second;
		std::list<std::pair<struct MaskState, unsigned> >::iterator itr2 = rec0->begin();
		std::list<unsigned long long>::iterator itr3 = rec1->begin();

		for(; itr2 != rec0->end() && itr3 != rec1->end(); itr2++, itr3++) {
			struct MaskState& maskstate = itr2->first;
			unsigned run_len = itr2->second;
			unsigned long long begintime = *itr3;
			fprintf(file, "%u\t%u\t%lu\t%u\t%llu\t%u\n",
					shader_id, warp_id, maskstate.the_mask, maskstate.count, begintime, run_len);
		}
	}
	fflush(file);
	fclose(file);
	printf("[Cartographer] Simt Stack history written to %s\n", file_name);
}

void Cartographer::writeWarpScheduledHistory(char* file_name) {
	assert(is_log_warp_sched_evts);
	FILE* file = fopen(file_name, "w");
	if(!file) file = stderr;
	fprintf(file, "CoreID\tWarpID\tCycle\tFU\n");
	WARP_SCHEDULED_TIME_Ty::iterator itr = m_warp_scheduled.begin();
	for(; itr != m_warp_scheduled.end(); itr++) {
		WARP_SCHEDULED_TIME_ENTRY_LIST_P_Ty rec0 = itr->second;
		unsigned shader_id = itr->first.first;
		unsigned warp_id   = itr->first.second;
		for(std::list<WARP_SCHEDULED_TIME_ENTRY>::iterator itr1 =
			rec0->begin(); itr1!=rec0->end(); itr1++) {
			unsigned long long cyc = itr1->first;
			char fu = itr1->second;
			fprintf(file, "%u\t%u\t%llu\t%d\n", shader_id, warp_id, cyc, fu);
		}
	}
	fflush(file);
	fclose(file);
	printf("[Cartographer] Warp being scheduled events written to %s\n", file_name);
}

void Cartographer::openAndInitializeCoalescedDB() {
	int err;
	sqlite3* db;
	if((err = sqlite3_open("address_coalescing.db", &db)) != SQLITE_OK) {
		assert(0 && "SQLite database open error");
	} else {
		// Create table
		sqlite3_stmt* drop_stmt;
		std::string drop_query = "DROP TABLE IF EXISTS koalesced;";
		sqlite3_prepare_v2(db, drop_query.c_str(), (int)(drop_query.size()),
			&drop_stmt, NULL);
		sqlite3_step(drop_stmt);
		sqlite3_finalize(drop_stmt);
	}
	sqlite3_close(db);
}

void Cartographer::linkCommittingTxnToWarp(SHADER_WARP_ID_Ty swid, MyCommittingTxnEntry* cety) {
	assert (m_live_tcommit_serial.find(swid) != m_live_tcommit_serial.end());
	unsigned serial = m_live_tcommit_serial.at(swid);
	cety->sandwich_sn = serial;
	if (m_linked_commit_txns.find(serial) == m_linked_commit_txns.end())
		m_linked_commit_txns[serial] = std::set<MyCommittingTxnEntry*>();
	m_linked_commit_txns.at(serial).insert(cety);
}

void Cartographer::summary() {
	printf("[Cartographer] Beginning of summary\n");

	unsigned total_l2_rm = 0, total_l2_wm = 0;
	std::map<unsigned, unsigned>::iterator itr;
	for(itr=l2_read_miss.begin(); itr!=l2_read_miss.end(); itr++) {
		unsigned d = itr->second;
		total_l2_rm += d;
	}
	for(itr=l2_write_miss.begin(); itr!=l2_write_miss.end(); itr++) {
		unsigned d = itr->second;
		total_l2_wm += d;
	}
	printf("[Cartographer total L2 R+W misses = %u + %u, sum=%u\n",
			total_l2_rm, total_l2_wm, total_l2_rm + total_l2_wm);

	std::map<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> >* lists[] = {&m_l1_misses, &m_l2_misses};
	unsigned sum_misses_r[] = {0, 0};
	unsigned sum_misses_w[] = {0, 0};
	for(int i=0; i<2; i++)
	{
		int sum_r = 0, sum_w = 0;
		std::map<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> >* curr = lists[i];
		for(std::map<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> >::iterator itr =
				curr->begin(); itr != curr->end(); itr++) {
			sum_w += itr->second.first;
			sum_r += itr->second.second;
		}
		sum_misses_r[i] = sum_r;
		sum_misses_w[i] = sum_w;
	}
	printf("Sum of L1 misses (R+W) = %u + %u = %u\n",sum_misses_r[0], sum_misses_w[0],
			sum_misses_r[0]+sum_misses_w[0]);
	printf("Sum of L2 misses (R+W) = %u + %u = %u\n",sum_misses_r[1], sum_misses_w[1],
			sum_misses_r[1]+sum_misses_w[1]);

	// Printing out SIMT stack in RLE form.

	// Print dangling guys

	for(std::map<SHADER_WARP_ID_Ty, std::list<InstLifeTimeInfo>* >::iterator itr = m_live_inst_lifetimes.begin();
			itr != m_live_inst_lifetimes.end(); itr++) {
		const SHADER_WARP_ID_Ty* swid = &(itr->first);
		const std::list<InstLifeTimeInfo>* warp_list = (itr->second);
		if(warp_list->size() > 0) {
			fprintf(stderr, "[Cartographer, LifeTime] Shader (%u,%u) has %lu instructions still on the fly, here they are:\n",
				swid->first, swid->second, warp_list->size());
			for(InstLifeTimeInfo ilti : *warp_list) {
				ilti.print_debug(stderr);
			}
			fprintf(stderr, "[Cartographer, LifeTime] end of list\n");
		}
	}

	dlopen("libsqlite3.so", RTLD_LAZY);

	// Print archived guys
	if(sandwich_db_filename != NULL) {
		writeArchivedInstLifetimeStats(sandwich_db_filename);
	} else {
		printf("[Cartographer] Not storing sandwich database.\n");
	}

	if(sandwich_plot_filename != NULL) {
		writeArchiveInstLifetimeStatsTxt(sandwich_plot_filename);
	} else {
		printf("[Cartographer] Not storing sandwich TEXT file\n");
	}

	if(is_log_coalesced_addrs) {
		dumpCurrentCoalescedToDB();
	}

	if (g_tommy_log_0902) { DumpTMHistory(); }

	if (is_log_simdfu_occupancy) { dumpDarkSiliconOpportunity(); }
}

void Cartographer::writeArchiveInstLifetimeStatsTxt(const char* filename) {
	FILE* file = fopen(filename, "w");

	fprintf(file, "CoreID\tWarpID\tInstruction\tFetch\tICacheAccessReady\tScheduled\tSchConstraintDelay\tFUStalled\tExecuted\tRetired\tFU\tSN\n");
	for(std::map<SHADER_WARP_ID_Ty, std::list<InstLifeTimeInfo>* >::iterator itr = m_done_inst_lifetimes.begin();
			itr != m_done_inst_lifetimes.end(); itr++) {
		const SHADER_WARP_ID_Ty& swid = (*itr).first;
		std::list<InstLifeTimeInfo>* the_ilti = (*itr).second;
		if(the_ilti && the_ilti->size() > 0) {
			printf("%lu records for warp (%u,%u)\n", the_ilti->size(), swid.first, swid.second);
			for(std::list<InstLifeTimeInfo>::iterator itr1 = the_ilti->begin(); itr1 != the_ilti->end(); itr1++) {
				const InstLifeTimeInfo& ilti = *itr1;
				fprintf(file, "%u\t%u\t%s\t%llu\t%u\t%llu\t%llu\t%llu\t%llu\t%llu\t%s\t%ld\n",
					swid.first, swid.second, ilti.inst_string.c_str(),
					ilti.fetch_began, ilti.icache_access_ready, ilti.scheduled,
					ilti.sched_num_issued_constraint_until,
					ilti.stalled_in_fu_until,
					ilti.executed, ilti.retired, ilti.func_unit,
					ilti.serial);
			}
		}
	}
}

void Cartographer::writeArchivedInstLifetimeStats(const char* filename) {
	fprintf(stderr, "[Cartographer] writeArchivedInstLifetimeStats(%s)\n", filename);
	int err;
	sqlite3* db;
	if((err=sqlite3_open(filename, &db)) != SQLITE_OK) {
		assert(0 && "SQLite database open error.\n");
	} else {
		fprintf(stderr, "SQLite database open OK\n");
	}
	// Create Table
	{
		sqlite3_stmt* drop_stmt;
		std::string drop_query = "DROP TABLE IF EXISTS sandwich;";
		err = sqlite3_prepare_v2(db, drop_query.c_str(), drop_query.size(),
			&drop_stmt, NULL);
		if(err != SQLITE_OK) {
			fprintf(stderr, "Error code = %d\n", err);
			assert(false && "Error dropping table");
		}
		sqlite3_finalize(drop_stmt);

		sqlite3_stmt* create_stmt;
		std::string create_query = "CREATE TABLE IF NOT EXISTS sandwich (CoreID INTEGER, WarpID INTEGER, Instruction TEXT, Fetch INTEGER,";
		create_query += " ICacheAccessReady INTEGER,";
		create_query += " Scheduled INTEGER, SchConstraintDelay INTEGER, FUStalled INTEGER, Executed INTEGER, ActiveMask INTEGER, Retired INTEGER, FU INTEGER, SN INTEGER,";
		create_query += " ScoreboardDeps TEXT, FUStalledBy TEXT, LDSTAccessReady INTEGER);";
		sqlite3_prepare_v2(db, create_query.c_str(), (int)(create_query.size()),
			&create_stmt, NULL);
		if(!(sqlite3_step(create_stmt) == SQLITE_DONE)) {
			fprintf(stderr, "Oh! Cannot create table!\n");
			return;
		}
		sqlite3_finalize(create_stmt);

		// "(SN, TIDInWarp, SendRS, SendWS, DoneFill, SendTxPassFail, DoneScoreboardCommit, "
		// "RSDetails, WSDetails, CUReplyDetails) "
		sqlite3_stmt* create_stmt1;
		const char* create_query1 = "CREATE TABLE IF NOT EXISTS sandwich_commit_tx "
				"(SN INTEGER, TIDInWarp INTEGER, SendRS INTEGER, SendWS INTEGER, DoneFill INTEGER, "
				"SendTxPassFail INTEGER, DoneScoreboardCommit INTEGER, "
				"RSDetails TEXT, WSDetails TEXT, CUReplyDetails TEXT) ";
		sqlite3_prepare_v2(db, create_query1, strlen(create_query1), &create_stmt1, NULL);
		if (!(sqlite3_step(create_stmt1) == SQLITE_DONE)) {
			fprintf(stderr, "Oh! Cannot create tabld!\n"); return;
		}
		sqlite3_finalize(create_stmt1);
	}

	// Begin and end transaction
	fprintf(stderr, "Begin database transaction\n");

	struct timeval tick, tock;
	unsigned long num_ety = 0, num_ety_ctxn = 0;
	gettimeofday(&tick, NULL);
	sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
	// Insert rows
	for(std::map<SHADER_WARP_ID_Ty, std::list<InstLifeTimeInfo>* >::iterator itr = m_done_inst_lifetimes.begin();
		itr != m_done_inst_lifetimes.end(); itr++) {
		const SHADER_WARP_ID_Ty& swid = (*itr).first;
		std::list<InstLifeTimeInfo>* the_ilti = (*itr).second;
		for(std::list<InstLifeTimeInfo>::iterator itr1 = the_ilti->begin(); itr1 != the_ilti->end(); itr1++) {
			InstLifeTimeInfo& ilti = *itr1;
			ilti.insertToSQLiteDB(db, swid.first, swid.second);
			num_ety++;
		}
	}

	if (linked_cartographerts) {
		std::vector<MyCommittingTxnEntry*>* txns = &(linked_cartographerts->txn_cid_done);
		for (std::vector<MyCommittingTxnEntry*>::iterator itr = txns->begin(); itr != txns->end(); itr++) {
			(*itr)->insertToSQLiteDB(db);
			num_ety_ctxn ++;
		}
	}

	sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
	gettimeofday(&tock, NULL);
	unsigned long tdiff = (tock.tv_sec - tick.tv_sec) * 1000 + (tock.tv_usec - tick.tv_usec) / 1000;
	fprintf(stderr, "Inserted %lu InstLifeTime entries and %lu commit tx entries in %lu milliseconds\n", num_ety,
			num_ety_ctxn, tdiff);
	fprintf(stderr, "End database transaction\n");
}

// -------------------------------------
// 2013-07-16: this routine needs much correction
// plus, it's not playing well with Dot now (may be too many edges)
// -------------------------------------

void Cartographer_PrintDummyFunctionInfoGraph(function_info* finfo, FILE* fh) {
	g_cartographer->PrintDummyFunctionInfoGraph(finfo, fh);
}
void Cartographer::PrintDummyFunctionInfoGraph(function_info* finfo, FILE* fh) {
	// Get information about the just-assembled PTX function info
	/*
	fprintf(fh, "********************8<************************\n");
	fprintf(fh, "[tommy] Function %s has %d instructions.\n",
			finfo->get_name().c_str(),
			finfo->get_function_size());
	fprintf(fh, "\n");
	*/

	fprintf(fh, "digraph %s {\n", finfo->get_name().c_str());
	fprintf(fh, "\tlabel=\"%s (%d instructions)\"\n", finfo->get_name().c_str(), finfo->get_function_size());
	fprintf(fh, "\tedge [penwidth=0.5]\n");

	// What registers do the instructions write to/read from?
	if(false)
	{
		for(std::list<ptx_instruction*>::iterator itr = finfo->m_instructions.begin();
				itr != finfo->m_instructions.end(); itr++) {
			ptx_instruction* ptx_ptr = *itr;
			ptx_ptr->print_insn(stdout);
			printf("\nout=[");
			for(int i=0; i<4; i++) printf("%d ", ptx_ptr->out[i]);
			printf("], in=[");
			for(int i=0; i<4; i++) printf("%d ", ptx_ptr->in[i]);
			printf("], arch.src=[");
			for(int i=0; i<8; i++) printf("%d ", ptx_ptr->arch_reg.src[i]);
			printf("], arch.dst=[");
			for(int i=0; i<8; i++) printf("%d ", ptx_ptr->arch_reg.dst[i]);
			printf("\n");
		}
	}

	// List all statements
	fprintf(fh, "\t{\n");
	fprintf(fh, "\t\tnode [shape=plaintext, fontsize=8];\n");
	std::vector<bool> is_inst_label;
	std::map<ptx_instruction*, int> inst_to_instid;
	std::list<ptx_instruction*>::iterator ptx_itr;
	int inst_id = 0;
	for (ptx_itr = finfo->m_instructions.begin();ptx_itr != finfo->m_instructions.end(); ptx_itr++, inst_id++) {
		ptx_instruction* ptx_ptr = *ptx_itr;
		is_inst_label.push_back(ptx_ptr->is_label());
		assert( ptx_ptr->get_bb()!=NULL );
	//	if(ptx_itr != finfo->m_instructions.begin()) { printf(" ->\n"); }
		fprintf(fh, "\t\tnode%d [label=\"", inst_id);
		(*ptx_itr)->print_insn(fh);
		fprintf(fh, "\"];\n");
		inst_to_instid[ptx_ptr] = inst_id;
	}
	fprintf(fh, "\n");
	fprintf(fh, "\t\t");
	for(int i=0; i<inst_id-1; i++) {
		fprintf(fh, "node%d -> ", i);
	}
	fprintf(fh, "node%d\n", inst_id-1);

	fprintf(fh, "\t}");

	fprintf(fh, "\t{ node [fontsize=8]; edge[penwidth=0.5];\n");
	const int NUM_PIPELINE_STAGES = 6;
	const char* pipeline_stages[] = {"fetch", "decode", "issue", "read_opr", "exec", "writeback"};
	for(int i=0; i<inst_id; i++) {
		if(is_inst_label[i] == true) continue;
		fprintf(fh, "\n\n\t\t{");
		fprintf(fh, "\t\t\trank = same;\n");
		fprintf(fh, "\t\t\tnode%d; ", i);
		for(int psidx = 0; psidx < NUM_PIPELINE_STAGES; psidx++) {
			fprintf(fh, "node%d_%d [label=\"%s\"]; ", i, psidx,
					pipeline_stages[psidx]);
		}
		fprintf(fh, "\t\t}\n");
	}
	fprintf(fh, "\t}\n");

	// Add vertical edges.
	{
		for(int stage=0; stage<NUM_PIPELINE_STAGES; stage++) {
			int inst_from = -1, inst_to = -1;
			fprintf(fh, "\tsubgraph cluster%d {\n", stage);
			if(stage!=0 && stage!=5) {
				fprintf(fh, "\t\tedge [style=invis]\n");
			}
			for(int i=0; i<inst_id; i++) {
				if(is_inst_label[i]==false) {
					inst_from = inst_to;
					inst_to = i;
				} else { continue; }
				if(inst_from != -1 && inst_to != -1) {
					// Fetch connected to fetch for in-order fetch
					// WB connected to WB for in-order retire

					fprintf(fh, "\t\tnode%d_%d -> node%d_%d\n",
						inst_from, stage, inst_to, stage);

				}
			}
			fprintf(fh, "\t}\n");
		}
	}

	// Add edges that correspond to pipeline stages.
	{
		for(int i=0; i<inst_id; i++) {
			if(is_inst_label[i] == false) {
				fprintf(fh, "\t");
				for(int psidx=0; psidx < NUM_PIPELINE_STAGES; psidx++) {
					fprintf(fh, "node%d_%d", i, psidx);
					if(psidx < NUM_PIPELINE_STAGES-1) fprintf(fh, " -> ");
				}
				fprintf(fh, "\n");
			}
		}
	}

	// Add edges that correspond to data dependency.
	// Limitations: currently only dependencies inside 1 BB is visualized.
	// Data dependency definition: from when the scoreboard releases a register
	//   to when the scoreboard checks for collisions
	{
		std::list<ptx_instruction*>::iterator ptx_itr, ptx_itr2;
		ptx_instruction* from, *to;
		for (ptx_itr = finfo->m_instructions.begin();ptx_itr != finfo->m_instructions.end(); ptx_itr++) {
			from = *ptx_itr;
			if(from->is_label()) continue;
			ptx_itr2 = ptx_itr;
//			bool clobbed = false;
			for(ptx_itr2++; ptx_itr2 != finfo->m_instructions.end(); ptx_itr2++) {
				to = *ptx_itr2;
				if(to->is_label()) continue;

				// Maybe sfu and sp has different latencies..... so they need different edge lengths/colors?
				// Assert: only 1 or 0 output register
				int has_dependency = false;
				const int NEG_ONE = (-1);
				std::set<int> dep_registers;
				for(unsigned iro = 0; iro < MAX_REG_OPERANDS; iro++) { // IRO = Index of Register_Output
					int from_reg_id = from->arch_reg.dst[iro];
					if(from_reg_id == NEG_ONE || from_reg_id==0) continue;
					int out_register = from_reg_id;
					for(unsigned iri = 0; iri < MAX_REG_OPERANDS; iri++) { // IRI = Index of Register_Input
						int to_reg_id = to->arch_reg.src[iri];
						if(to_reg_id == NEG_ONE || to_reg_id == 0) continue;
						int in_register = to_reg_id;
						if(out_register == in_register) {
							has_dependency = true;
							dep_registers.insert(out_register);
							break;
						}
					}
				}

				// Clobbed by the next inst?
				for(unsigned iro=0; iro < MAX_REG_OPERANDS; iro++) {
					int clobber = to->arch_reg.dst[iro];
					if(clobber == NEG_ONE || clobber == 0) continue;
					if(clobber == from->arch_reg.dst[iro]) {
						assert(iro == 0 || iro == 1); // Should only have 1 output register.
//						clobbed = true;
						break;
					}
				}
				if(has_dependency) {
					for(std::set<int>::iterator itr = dep_registers.begin(); itr != dep_registers.end(); itr++) {
					fprintf(fh, "\tnode%d_%d -> node%d_%d [constraint=false, label=\"%d\"]\n",
							inst_to_instid[from], 5, // 5 = writeback where the scoreboard releases.
							inst_to_instid[to], 2,
							*itr);  // 2 = issue, where scoreboard checks collisions
					}
				}
			}
		}
	}

	/*
		writeback();
		execute();
		read_operands();
		issue();
		decode();
		fetch();
		*/
	fprintf(fh, "}\n");
	fflush(fh);
}

// Moved here to reduce rebuild time
class InstInPipeline {
friend class Cartographer;
public:
	InstInPipeline(unsigned _pc, mem_fetch* _mem_fetch,
			unsigned long long _last_fetch):
		m_pc(_pc), m_mem_fetch(_mem_fetch), m_last_fetch(_last_fetch),
		m_fetch_buffer_filled(-1), m_decoded(0) {
		m_warp_inst = NULL;
	}
private:
	unsigned m_pc;
	mem_fetch* m_mem_fetch;
	const warp_inst_t* m_warp_inst;
	unsigned long long m_last_fetch;
	unsigned long long m_fetch_buffer_filled; // When did inst$ service complete/hit occur?
	unsigned long long m_decoded;             // When is this instruction decoded?
};


//
// Only log fetch event from Shader 0 Warp 0
//
const int TRK_STATUS_FETCH = -999,
		   TRK_STATUS_DECODE = 20;

void Cartographer::On_shader_core_fetch_L1I_miss(shader_core_ctx* from, mem_fetch* mf) {
	if(!ENABLE_CARTOGRAPHER) return;
	int sid = mf->m_sid;
	int warp_id = mf->m_wid;
	unsigned pc = mf->m_access.m_addr - 0xF0000000;
	if(!(sid==0 && warp_id==0)) return;
	unsigned long long int ctx_last_fetch = gpu_sim_cycle;
	// Push into watch list (or say the list of tracked instructions).
	InstInPipeline* iipl = new InstInPipeline(pc, mf, ctx_last_fetch);
	tracked_insts.push_back(iipl);
	printf("[tommy] OnShaderCoreFetch I$ miss Created core 0 warp 0, inst PC=%#x, at cycle=%llu\n",
			pc, gpu_tot_sim_cycle + gpu_sim_cycle);
}

void Cartographer::On_shader_core_fetch_L1I_hit(shader_core_ctx* from, mem_fetch* mf) {
	if(!ENABLE_CARTOGRAPHER) return;
	int sid = mf->m_sid;
	int warp_id = mf->m_wid;
	unsigned pc = mf->m_access.m_addr - 0xF0000000;
	if(!(sid==0 && warp_id==0)) return;
	unsigned long long int ctx_last_fetch = gpu_sim_cycle;
	InstInPipeline* iipl = new InstInPipeline(pc, mf, ctx_last_fetch);
	iipl->m_fetch_buffer_filled = ctx_last_fetch;
	tracked_insts.push_back(iipl);
	printf("[tommy] OnShaderCoreFetch I$ hit Created core 0 warp 0, inst PC=%#x, at cycle=%llu\n",
			pc, gpu_tot_sim_cycle + gpu_sim_cycle);
}

void Cartographer::On_shader_core_decode(shader_core_ctx* from, const warp_inst_t* inst) {
	if(!ENABLE_CARTOGRAPHER) return;
	struct ifetch_buffer_t* fetch_buffer = &(from->m_inst_fetch_buffer);
	int warp_id = fetch_buffer->m_warp_id;
	if(from->m_sid != 0) return;
	if(warp_id != 0) return;
	unsigned pc = fetch_buffer->m_pc;

	// The instruction's last_fetch must be accessible through the shader core context
	unsigned long long int ctx_last_fetch = from->m_warp[0].m_last_fetch;
	InstInPipeline* tracked_inst = NULL;
	{
		bool is_found = false;
		for(std::list<InstInPipeline*>::iterator itr = tracked_insts.begin();
				itr != tracked_insts.end(); itr++) {
			InstInPipeline* p = *itr;
			if(p->m_last_fetch == ctx_last_fetch) {
				tracked_inst = p; is_found = true; break;
			}
		}
		if(!is_found) {
			assert(0 && "Ah! Oh! No!");
		}
	}

	// Note: 1 memory fetch could result in 2 instructions.
	if(tracked_inst->m_warp_inst == NULL) {
		printf("[tommy] OnShaderCoreDecode core 0 warp 0 inst PC=%#x, at cycle=%llu\n",
				pc, gpu_tot_sim_cycle + gpu_sim_cycle);
		tracked_inst->m_warp_inst = inst;
	} else {
		InstInPipeline* the_brother_inst = new InstInPipeline(*tracked_inst);
		printf("[tommy] OnShaderCoreDecode[2] core 0 warp 0 inst PC=%#x, at cycle=%llu\n",
				pc, gpu_tot_sim_cycle + gpu_sim_cycle);
		the_brother_inst->m_warp_inst = inst;
	}
}
//
//static int new_serial_number = 0;
//int Cartographer::getNextFreeSerialNumberAllWarps(shader_core_ctx* shader) {
//	for(int x=0; x < 32767; x++) {
//		new_serial_number ++;
//		if(new_serial_number >= 32767) new_serial_number = 0;
//		bool is_ok = true;
//		for(unsigned wid=0; wid < shader->m_config->max_warps_per_shader; wid++) {
//			SHADER_WARP_ID_Ty swid(shader->get_sid(), wid);
//			if(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end()) {
//				std::map<int, InstLifeTimeInfo>* warp_map = m_live_inst_lifetimes[swid];
//				if(warp_map->find(new_serial_number)!=warp_map->end()) {
//					is_ok = false; break;
//				}
//			}
//		}
//		if(is_ok) return new_serial_number;
//	}
//	return -999;
//}

void Cartographer::printLiveInstStats(SHADER_WARP_ID_Ty& swid) {
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	std::list<InstLifeTimeInfo>* the_warp_list = m_live_inst_lifetimes[swid];
	fprintf(stderr, "Warp [%u,%u]\n", swid.first, swid.second);
	for(std::list<InstLifeTimeInfo>::iterator itr1 = the_warp_list->begin();
				itr1 != the_warp_list->end(); itr1++) {
		(*itr1).print(stderr);
	}
}

void Cartographer::printLiveInstStats1(unsigned sid, unsigned wid) {
	SHADER_WARP_ID_Ty swid = std::make_pair(sid, wid);
	printLiveInstStats(swid);
}

int Cartographer::getNextFreeSerialNumberOneWarp(SHADER_WARP_ID_Ty swid) {
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	if (curr_max_sn.find(swid) == curr_max_sn.end()) {
		curr_max_sn[swid] = 0;
	}
	curr_max_sn.at(swid) += 1;
	return curr_max_sn.at(swid);
}

// Do not use this on Fetch
InstLifeTimeInfo* Cartographer::findWarpInstAtPipelineStageEX(SHADER_WARP_ID_Ty& swid, const warp_inst_t* _inst, unsigned char stage) {
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	std::list<InstLifeTimeInfo>* the_warp_list = m_live_inst_lifetimes[swid];
	InstLifeTimeInfo* ret = NULL;
	unsigned num_found = 0;
	// Asserting that one instruction cannot be decoded twice.
	for(std::list<InstLifeTimeInfo>::iterator itr = the_warp_list->begin();
		itr != the_warp_list->end(); itr++) {
		if(itr->stage == stage && (itr->inst == _inst)) {
			ret = &(*itr);
			num_found ++;
		}
	}

	if(num_found == 0) {
		for(std::list<InstLifeTimeInfo>::iterator itr = the_warp_list->begin();
			itr != the_warp_list->end(); itr++) {
			if(itr->stage == stage && (itr->inst->pc == _inst->pc)) { // Ugly fix (Comparing PC): Sep 02
				ret = &(*itr);
				num_found ++;
			}
		}
	}

	if(!(num_found == 1 || num_found == 0)) {
		fprintf(stderr, "[!FindWarpInstAtPPLStageEx multiple results! (%u,%u) inst=%p stage=%d\n",
				swid.first, swid.second, _inst, stage);
		printLiveInstStats(swid);
		assert(0);
	}
	return ret;
}

// Even searching in all warps there shall be only 1 match
InstLifeTimeInfo* Cartographer::findWarpInstAtPipelineStageEXAllWarps(unsigned shader_id, const warp_inst_t* _inst, unsigned char stage) {
	InstLifeTimeInfo* ret = NULL;
	unsigned num_found = 0;
	for(unsigned wid=0; wid < num_warps_per_shader; wid++) {
		SHADER_WARP_ID_Ty swid(shader_id, wid);
		std::map<SHADER_WARP_ID_Ty, std::list<InstLifeTimeInfo>* >::iterator itr = m_live_inst_lifetimes.find(swid);
		if(itr == m_live_inst_lifetimes.end()) continue;
		std::list<InstLifeTimeInfo>* the_warp_list = itr->second;
		for(std::list<InstLifeTimeInfo>::iterator itr2 = the_warp_list->begin(); itr2 != the_warp_list->end(); itr2++) {
			if((*itr2).stage == stage && (*itr2).inst == _inst) {
				ret = &(*itr2);
				num_found++;
			}
		}
	}
	assert(num_found==0 || num_found==1);
	return ret;
}

InstLifeTimeInfo* Cartographer::findWarpInstAtPipelineStage(SHADER_WARP_ID_Ty& swid, unsigned char stage) {
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	std::list<InstLifeTimeInfo>* the_warp_list = m_live_inst_lifetimes[swid];
	InstLifeTimeInfo* ret = NULL;
	unsigned num_found = 0;
	// There may be 2 instructions at the same stage: Decode
	for(std::list<InstLifeTimeInfo>::iterator itr = the_warp_list->begin();
			itr != the_warp_list->end(); itr++) {
		if((*itr).stage == stage) {
			num_found ++;
//			(*itr).print(stderr);
			ret = (&*itr);
		}
	}
	return ret;
}

// Called on fetch()
void Cartographer_onNewInstructionBorn(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf, bool is_missed) {
	return g_cartographer->onNewInstructionBorn(shader, warp_id, mf, is_missed);
}
void Cartographer::onNewInstructionBorn(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf, bool is_missed) {
	unsigned shader_id = shader->get_sid();
	if(!shouldLog(shader_id, warp_id)) return;


	// Handle empty list
	SHADER_WARP_ID_Ty swid = std::make_pair(shader_id, warp_id);
	if(m_live_inst_lifetimes.find(swid) == m_live_inst_lifetimes.end()) {
		std::list<InstLifeTimeInfo>* new_warp_list = new std::list<InstLifeTimeInfo>();
		m_live_inst_lifetimes[swid] = new_warp_list;
	}
	std::list<InstLifeTimeInfo>* the_warp_list = m_live_inst_lifetimes[swid];

	address_type this_pc = mf->m_access.m_addr;
	DBG(printf("[Cartographer::onNewInstructionBorn] (%d,%d), PC=%X\n", shader->get_sid(), warp_id, this_pc));

	InstLifeTimeInfo* fetch0 = findWarpInstAtPipelineStage(swid, PPL_AFTER_FETCH);
	if(fetch0) { // 20130813 Bugfix
		if(this_pc != fetch0->pc) {
			// Tommy: fetch0 has become invalid as of now.
			//        Observed in wave, blackscholes, vectoradd
			//
			if(0)
				fprintf(stderr, "! PC not matched, %x vs %x\n", this_pc, fetch0->pc);
//			fetch0->print_debug(stderr);
//			assert(0);
			// 20130813 Ugly bugfix: delete fetch0
			the_warp_list->remove(*fetch0);
		} else {
			return;
		}
	}

	InstLifeTimeInfo new_info;
	new_info.fetch_began = getCurrentCycleCount();
	new_info.pc = this_pc;

	assert ((this_pc & 0xF0000000) == 0xF0000000);

	new_info.stage = PPL_AFTER_FETCH;
	new_info.is_fetch_missed = is_missed;
	if (is_missed == false)
		new_info.icache_access_ready = new_info.fetch_began;
	(*the_warp_list).push_back(new_info);
}

void Cartographer_onIcacheAccessReady(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf) {
	g_cartographer->onICacheAccessReady(shader, warp_id, mf);
}
void Cartographer::onICacheAccessReady(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf) {
	unsigned shader_id = shader->get_sid();
	SHADER_WARP_ID_Ty swid = std::make_pair(shader_id, warp_id);
	if (!shouldLog(shader_id, warp_id)) return;

	//	InstLifeTimeInfo* fetch = findWarpInstAtPipelineStage(swid, PPL_AFTER_FETCH);
	// Need to also check PC so cannot use findWarpInstAtPipelineStage

	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	std::list<InstLifeTimeInfo>* the_warp_list = m_live_inst_lifetimes[swid];
	InstLifeTimeInfo* fetch = NULL;
	unsigned num_found = 0;
	// There may be 2 instructions at the same stage: Decode
	for(std::list<InstLifeTimeInfo>::iterator itr = the_warp_list->begin();
			itr != the_warp_list->end(); itr++) {
		if((*itr).stage == PPL_AFTER_FETCH && (*itr).pc == mf->m_access.m_addr) {
			num_found ++;
			fetch = (&*itr);
		}
	}

	assert(fetch);
	// Must be in "Live Instructions"
	std::list<InstLifeTimeInfo>* the_list = m_live_inst_lifetimes.at(swid);
	for (std::list<InstLifeTimeInfo>::iterator itr = the_list->begin(); itr != the_list->end();
			itr ++) {
		const InstLifeTimeInfo& x = (*itr);
		if (x.shader_id == shader_id && x.warp_id == warp_id && x.pc == mf->m_access.m_addr) {
			if (&x != fetch) {

				printf("There seem to be > 1 InstLifeTimeInfo entry in the current list.\n");
				for (std::list<InstLifeTimeInfo>::iterator itr2 = the_list->begin(); itr2 != the_list->end();
					itr2 ++) {
					if (&x == &(*itr2)) printf(" >> "); else printf("    ");
					itr2->print(stdout);
				}

				printf("&x=%p, fetch=%p\n", &x, fetch);
				assert(0);
			}
			break;
		}
	}

	assert(fetch->is_fetch_missed == true);
	fetch->icache_access_ready = getCurrentCycleCount();
}

void Cartographer_onInstructionsDecoded(shader_core_ctx* shader,
		unsigned warp_id, warp_inst_t* inst1, warp_inst_t* inst2) {
	g_cartographer->onInstructionsDecoded(shader, warp_id, inst1, inst2);
}

// 20130812: Bug on vectorAdd
// inst2 == 0x200000005 (?!)
void Cartographer::onInstructionsDecoded(shader_core_ctx* shader,
		unsigned warp_id, warp_inst_t* inst1, warp_inst_t* inst2) {

	if(!shouldLog(shader->get_sid(), warp_id)) return;

	SHADER_WARP_ID_Ty swid = std::make_pair(shader->get_sid(), warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	std::list<InstLifeTimeInfo>* warp_list = m_live_inst_lifetimes[swid];

//	fprintf(stderr, "%u, %u\n", shader->get_sid(), warp_id);
	InstLifeTimeInfo* the_fetch = findWarpInstAtPipelineStage(swid, PPL_AFTER_FETCH);
	assert(the_fetch);

	// Safe here because the warp ID is the ID of the warp who initiated the memory fetch
	unsigned long long the_fetch_began = the_fetch->fetch_began;

	warp_inst_t* insts[] = {inst1, inst2};
	for(unsigned i=0; i<sizeof(insts)/sizeof(warp_inst_t*); i++) {
		warp_inst_t* curr = insts[i];
		if(curr) {
			// Do not repeatedly insert entries
			InstLifeTimeInfo* ilti = findWarpInstAtPipelineStageEX(swid, curr, PPL_AFTER_DECODE);
			if(ilti) continue;

			// Support ptx instructions first...?
			std::string the_inst_string;
			std::string the_inst_string1;
			ptx_instruction* ptxinst = dynamic_cast<ptx_instruction*>(curr); // if we don't dyn cast, it will print "111111111111111111111111111111"
			if(ptxinst) {
				the_inst_string = ptxinst->to_string();
				the_inst_string1 = ptxinst->get_source();
			}
			else {
				the_inst_string = "(Not a PTX instruction)";
				the_inst_string1 = "(Not a PTX instruction)";
			}

			ptx_instruction* ptxI = dynamic_cast<ptx_instruction*>(curr);
			InstLifeTimeInfo new_info(*the_fetch);
			new_info.getNewSerialID();
			new_info.inst = curr;
			new_info.inst_string = the_inst_string;
			new_info.fetch_began = the_fetch_began;
			new_info.stage = PPL_AFTER_DECODE; // 1 means after decoding & waiting to be issued.
			int new_sn = getNextFreeSerialNumberOneWarp(swid);
			new_info.sn = new_sn;
			new_info.sched_num_issued_constraint_until = getCurrentCycleCount();
			new_info.ldst_access_ready = 0;
			if(ptxI) {
				new_info.pc = ptxI->get_PC();
				g_pc_to_ptxsource[new_info.pc] = the_inst_string;
				if (ptxI->is_tcommit) { new_info.is_tcommit = true; }
			}

//			printf("(%u, %u) Instruction %s decoded at cycle %llu\n",
//				swid.first, swid.second, the_inst_string1.c_str(), getCurrentCycleCount());

			curr->m_serial_number = new_sn; // Used for later pipeline stages
			DBG(printf("[Cartographer::decode] (%d,%d) opcode=%d pc=%d, sn=%d, listlen=%u\n",
				shader->get_sid(), warp_id,
				(int)(curr->op),
				curr->pc,
				new_sn,
				(*warp_list).size()));

//			assert (curr->pc != NULL); // When using arch=sm_11 the PC could be zero.
			(*warp_list).push_back(new_info);

			if(the_fetch->is_fetch_missed) {
				if(ptxI) {
					addr_t PC = ptxI->get_PC();
					if(edges_fetch[shader->get_sid()].find(PC) == edges_fetch[shader->get_sid()].end())
						edges_fetch[shader->get_sid()][PC] = new_info;
				}
			}
		} else return; // Some invalid things are passed in here.
	}

	unsigned long old_size = warp_list->size();
	warp_list->remove(*the_fetch);
	assert(warp_list->size() == old_size - 1);
}

void Cartographer_onInstructionDemises(shader_core_ctx* shader, unsigned warp_id, warp_inst_t* inst, unsigned char infoidx) {
	g_cartographer->onInstructionDemises(shader, warp_id, inst, infoidx);
}
void Cartographer::onInstructionDemises(shader_core_ctx* shader, unsigned warp_id, warp_inst_t* inst, unsigned char infoidx) {
	if(!shouldLog(shader->get_sid(), warp_id)) return;

	SHADER_WARP_ID_Ty swid = std::make_pair(shader->get_sid(), warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	std::list<InstLifeTimeInfo>* warp_list = m_live_inst_lifetimes[swid];

	DBG(printf("[Cartographer::onInstructionDemises] (%d,%d) SN=%d opcode=%d pc=%d\n",
		swid.first, swid.second, sn, inst->op, inst->pc));

	InstLifeTimeInfo* to_demise = NULL;
	if(infoidx == 0) to_demise = findWarpInstAtPipelineStageEX(swid, inst, PPL_AFTER_ISSUE_TO_FU);
	else if(infoidx == 1) to_demise = findWarpInstAtPipelineStageEX(swid, inst, PPL_AFTER_DECODE);
	if(!to_demise)
		to_demise = findWarpInstAtPipelineStageEX(swid, inst, PPL_AFTER_ISSUE);
	if(!to_demise)
		to_demise = findWarpInstAtPipelineStageEX(swid, inst, PPL_LDST_READY); // 2015-07-10 fix
	if(!to_demise) {
		fprintf(stderr, "%p\n", inst);
		ptx_instruction* static_inst = dynamic_cast<ptx_instruction*>(inst);
		if(static_inst) {
			static_inst->print_insn(stderr);
		} else {
	//		fprintf(stderr, "This instruction's static information is lost (inst-%p, swid=(%u,%u), stage=%d).\n",
	//				inst, swid.first, swid.second, PPL_AFTER_ISSUE_TO_FU);
		}
		printLiveInstStats(swid);
		assert(0);
	}
	if(!(to_demise->inst->pc == inst->pc)) {
		fprintf(stderr, "[! in demise], PC: %x vs %x\n", to_demise->inst->pc, inst->pc);
		inst->print_insn(stderr);
		fprintf(stderr, "\n");
		to_demise->inst->print_insn(stderr);
		fprintf(stderr, "\n");
		assert(0);
	}

	// Archive instruction lifetime analysis data
	if(m_done_inst_lifetimes.find(swid) == m_done_inst_lifetimes.end()) {
		m_done_inst_lifetimes[swid] = new std::list<InstLifeTimeInfo>();
	}
	std::list<InstLifeTimeInfo>* warp_iftilist = m_done_inst_lifetimes[swid];
	InstLifeTimeInfo the_demised(*to_demise);
	if(infoidx == 1) {
		if(!(the_demised.executed==0 && the_demised.scheduled==0)) {
//			fprintf(stderr, "%llu, %llu\n", the_demised.executed == 0, the_demised.scheduled == 0);
//			assert(the_demised.executed == 0 && the_demised.scheduled == 0);
		}
		the_demised.executed = the_demised.scheduled = the_demised.stalled_in_fu_until = getCurrentCycleCount();
		the_demised.func_unit = (char*)("Flushed");
	}
	if(inst->is_load() || inst->is_store()) { // every new object has a new serial id
		ldst_serials[inst] = the_demised.serial;
	}
	the_demised.retired = getCurrentCycleCount();
	warp_iftilist->push_back(the_demised);

	warp_list->remove(*to_demise);
}

void Cartographer_onLDSTAccessReady(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf) {
	g_cartographer->onLDSTAccessReady(shader, warp_id, mf);
}

void Cartographer::onLDSTAccessReady(shader_core_ctx* shader, unsigned warp_id, const mem_fetch* mf) {
	if(!shouldLog(shader->get_sid(), warp_id)) return;
	DBG(printf("[Cartographer::onLDSTAccessReady] (%d,%d) SN=%d\n",
		shader->get_sid(), warp_id, mf->m_serial_number));
	SHADER_WARP_ID_Ty swid = std::make_pair(shader->get_sid(), warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	std::list<InstLifeTimeInfo>* warp_list = m_live_inst_lifetimes.at(swid);
	InstLifeTimeInfo* inst = NULL;
	for (std::list<InstLifeTimeInfo>::iterator itr = warp_list->begin();
			itr != warp_list->end(); itr++) {
		if ( (*itr).sn == mf->m_serial_number) {
			inst = &(*itr);
			break;
		}
	}
	if (inst == NULL) {
		warp_list = m_done_inst_lifetimes.at(swid);
		for (std::list<InstLifeTimeInfo>::iterator itr = warp_list->begin();
					itr != warp_list->end(); itr++) {
			if ( (*itr).sn == mf->m_serial_number) {
				inst = &(*itr);
				break;
			}
		}
	}

	if (inst == NULL) {
		{
			std::list<InstLifeTimeInfo>* warp_list = m_live_inst_lifetimes.at(swid);
			printf("Live inst %p:\n", warp_list);
			for (std::list<InstLifeTimeInfo>::iterator itr = warp_list->begin();
						itr != warp_list->end(); itr++) {
				printf("PC=%d, SN=%d\n", itr->pc, itr->sn);
			}
		}
		{
			std::list<InstLifeTimeInfo>* warp_inst = m_done_inst_lifetimes.at(swid);
			printf("Done inst %p:\n", warp_inst);
			for (std::list<InstLifeTimeInfo>::iterator itr = warp_list->begin();
						itr != warp_list->end(); itr++) {
				printf("PC=%d, SN=%d\n", itr->pc, itr->sn);
			}
		}
		assert(0);
	}

	assert (inst != NULL);


//	if ( !(inst->pc == inst1->pc) )
	if (inst->stage != PPL_LDST_READY) {
		inst->ldst_access_ready = getCurrentCycleCount();
		inst->stage = PPL_LDST_READY;
	}
}

void Cartographer_onInstructionIssued(shader_core_ctx* shader, unsigned warp_id,
		const warp_inst_t* static_inst, warp_inst_t* dynamic_inst,
		const char* fu, const active_mask_t* amask) {
	g_cartographer->onInstructionIssued(shader, warp_id, static_inst, dynamic_inst, fu, amask);
}
void Cartographer::onInstructionIssued(shader_core_ctx* shader, unsigned warp_id,
		const warp_inst_t* static_inst, warp_inst_t* dynamic_inst,
		const char* fu, const active_mask_t* amask) {
	unsigned shader_id = shader->get_sid();

	if(!shouldLog(shader_id, warp_id)) return;
	SHADER_WARP_ID_Ty swid = std::make_pair(shader->get_sid(), warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());


	InstLifeTimeInfo* to_issue = findWarpInstAtPipelineStageEX(swid, static_inst, PPL_AFTER_DECODE);

	DBG(printf("[Cartographer::onInstructionIssued] (%d,%d) op=%d,%d sn=%d,%d, ilti's SN is %d, ldstready=%llu\n",
			shader_id, warp_id,
			dynamic_inst->op, static_inst->op,
			dynamic_inst->m_serial_number, static_inst->m_serial_number,
			to_issue->sn,
			to_issue->ldst_access_ready));
//	assert (to_issue->sn == dynamic_inst->m_serial_number);

	if((!to_issue) || (to_issue->inst != static_inst)) {
		if(to_issue) {
			to_issue->print(stderr);
		} else {
			fprintf(stderr, "to_issue is NULL\n");
		}
		fprintf(stderr, "%p %p %p\n", to_issue, to_issue->inst, static_inst);
		const ptx_instruction* p = dynamic_cast<const ptx_instruction*>(static_inst);
		if(p) {
			p->print_insn(stderr);
		}

		assert(to_issue && to_issue->inst == static_inst); // What does it become of ???
	}

	if(0) {
		std::string sz_inst;
		ptx_instruction* pi = dynamic_cast<ptx_instruction*>(to_issue->inst);
		if(pi) sz_inst = pi->get_source();
		else sz_inst = "(No PTX source)";
		fprintf(stderr, "(%u, %u) Instruction %s issued at cycle %llu to func unit %s, (%p->%p)\n",
			shader_id, warp_id,
			sz_inst.c_str(),
			getCurrentCycleCount(), fu,
			to_issue->inst, dynamic_inst);
	}

	to_issue->inst = (warp_inst_t*)dynamic_inst;
	to_issue->stage = PPL_AFTER_ISSUE;
	to_issue->func_unit = (char*)fu;
	to_issue->scheduled = getCurrentCycleCount();
	to_issue->stalled_in_fu_until = to_issue->scheduled;
	to_issue->active_mask = *amask;
	dynamic_inst->m_serial_number = to_issue->sn;

	if (dynamic_inst->is_tcommit) { // Cannot use "to_issue->is_tcommit" (will always return true)
		assert (m_live_tcommit_serial.find(swid) == m_live_tcommit_serial.end());
		m_live_tcommit_serial[swid] = to_issue->serial;
		tcommit_issued_serials.insert(to_issue->serial);
	}
}

// "Deps" will be non-empty only when scheduler detects scoreboard collisions!
void Cartographer_onInstructionIssueCountConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst, const std::set<warp_inst_t*>& deps) {
	g_cartographer->onInstructionIssueCountConstrained(shader, warp_id, s_inst, deps);
}
void Cartographer::onInstructionIssueCountConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst, const std::set<warp_inst_t*>& deps) {
	unsigned shader_id = shader->get_sid();
	if(!shouldLog(shader_id, warp_id)) return;
	SHADER_WARP_ID_Ty swid = std::make_pair(shader_id, warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());

	InstLifeTimeInfo* constrained = findWarpInstAtPipelineStageEX(swid, s_inst, PPL_AFTER_DECODE);
		assert(constrained && constrained->inst == s_inst);
//	fprintf(stderr, "This inst has been constrained. (%u, %u)\n", shader_id, warp_id);
	constrained->sched_num_issued_constraint_until = getCurrentCycleCount();

	// Deps logging.
	{
		std::set<warp_inst_t*>::const_iterator itr;
		for(itr = deps.begin(); itr != deps.end(); itr++) {
			const warp_inst_t* inst = *itr;
			InstLifeTimeInfo* dep = findWarpInstAtPipelineStageEX(swid, inst, PPL_AFTER_DECODE);
			long serial = 0;//unsigned long* new_instcnt_1 = (unsigned long*)(malloc(sizeof(unsigned long) * num_wp));
			if(!dep) {
				dep = findWarpInstAtPipelineStageEX(swid, inst, PPL_AFTER_ISSUE);
				if(!dep) {
					dep = findWarpInstAtPipelineStageEX(swid, inst, PPL_AFTER_ISSUE_TO_FU);
					if(!dep) {
						serial = ldst_serials[inst];
						goto ok_sn;
					}
				}
			}
			assert(dep);
			serial = dep->serial;
ok_sn:
			constrained->scoreboard_deps.insert(serial); // The serial won't change after 1 InstLifeTimeInfo has been decoded!!!
		}
	}
}

void Cartographer_onInstructionIssuedToFU(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst, const warp_inst_t* to_inst) {
	g_cartographer->onInstructionIssuedToFU(shader, warp_id, from_inst, to_inst);
}
void Cartographer::onInstructionIssuedToFU(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst, const warp_inst_t* to_inst) {
	unsigned shader_id = shader->get_sid();
	if(!shouldLog(shader_id, warp_id)) return;//unsigned long* new_instcnt_1 = (unsigned long*)(malloc(sizeof(unsigned long) * num_wp));
	SHADER_WARP_ID_Ty swid = std::make_pair(shader->get_sid(), warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	InstLifeTimeInfo* to_fu = findWarpInstAtPipelineStageEX(swid, from_inst, PPL_AFTER_ISSUE);
	assert(to_fu && to_fu->inst == from_inst);
	to_fu->inst = (warp_inst_t*)(to_inst);
	to_fu->stage= PPL_AFTER_ISSUE_TO_FU;
	to_fu->executed = getCurrentCycleCount();

	if(to_inst->is_load() || to_inst->is_store()) {
		ldst_serials[to_inst] = to_fu->serial;
		ldst_serials[from_inst] = to_fu->serial;
	}
	// Add execute time
}

void Cartographer_onInstructionFUConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst) {
	g_cartographer->onInstructionFUConstrained(shader, warp_id, from_inst);
}
void Cartographer::onInstructionFUConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from_inst) {
	unsigned shader_id = shader->get_sid();
	if(!shouldLog(shader_id, warp_id)) return;
	SHADER_WARP_ID_Ty swid = std::make_pair(shader->get_sid(), warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	InstLifeTimeInfo* the_stalled = findWarpInstAtPipelineStageEX(swid, from_inst, PPL_AFTER_ISSUE);
	if(the_stalled == NULL) {
		fprintf(stderr, "! the_stalled is NULL.\n");
		from_inst->print_insn(stderr);
		const ptx_instruction* pi = dynamic_cast<const ptx_instruction*>(from_inst);
		pi->print_insn(stderr);
	}
	assert(the_stalled->inst == from_inst);
	the_stalled->stalled_in_fu_until = getCurrentCycleCount();
}

void Cartographer_onMoveWarp(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from, const warp_inst_t* to, unsigned char stage) {
	g_cartographer->onMoveWarp(shader, warp_id, from, to, stage);
}
void Cartographer::onMoveWarp(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* from, const warp_inst_t* to, unsigned char stage) {
	unsigned shader_id = shader->get_sid();
	if(!shouldLog(shader_id, warp_id)) return;
	SHADER_WARP_ID_Ty swid = std::make_pair(shader_id, warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end());
	InstLifeTimeInfo* to_change = findWarpInstAtPipelineStageEX(swid, from, stage);
	assert(to_change);
	to_change->inst = (warp_inst_t*)to;
}


void Cartographer_onInstructionIssueResourceConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst,
		std::vector<warp_inst_t*>* mem, std::vector<warp_inst_t*>* sp, std::vector<warp_inst_t*>* sfu,
		bool memfree, bool spfree, bool sfufree) {
	g_cartographer->onInstructionIssueResourceConstrained(shader, warp_id, s_inst, mem, sp, sfu, memfree, spfree, sfufree);
}
void Cartographer::onInstructionIssueResourceConstrained(shader_core_ctx* shader, unsigned warp_id, const warp_inst_t* s_inst,
		std::vector<warp_inst_t*>* mem, std::vector<warp_inst_t*>* sp, std::vector<warp_inst_t*>* sfu,
		bool memfree, bool spfree, bool sfufree) {
	unsigned shader_id = shader->get_sid();
	if(!shouldLog(shader_id, warp_id)) return;
	SHADER_WARP_ID_Ty swid = std::make_pair(shader_id, warp_id);
	assert(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end()); // This loox like a mantra!
	InstLifeTimeInfo* stalled = findWarpInstAtPipelineStageEX(swid, s_inst, PPL_AFTER_DECODE);

	#define SHOW_ME_ALL_INSTS \
	for(unsigned wid = 0; wid < num_warps_per_shader; wid++) { \
		SHADER_WARP_ID_Ty swid(shader_id, wid); \
		printLiveInstStats(swid); \
	}

	if(!stalled) {
		const ptx_instruction* p = dynamic_cast<const ptx_instruction*>(s_inst);
		if(p) {
			p->print_insn(stderr);
		} else s_inst->print_insn(stderr);
		SHOW_ME_ALL_INSTS;
		assert(stalled != NULL);
	}

	// When an instruction is "issued to F.U.", it is not in the dispatch port anymore!
	//   that's why we only have to search for warp insts at pipeline stage PPL_AFTER_ISSUE
	//   rather than PPL_AFTER_ISSUE_TO_FP.

	std::vector<warp_inst_t*>::iterator itr;
	if(!memfree) {
		for(itr = mem->begin(); itr != mem->end(); itr++) {
			warp_inst_t* inst = *itr;
			assert(inst != s_inst);
			InstLifeTimeInfo* ilti = findWarpInstAtPipelineStageEXAllWarps(shader_id, inst, PPL_AFTER_ISSUE);
			if(ilti == NULL) {
				printf("insts %p blocked by %p on MEM port.\n", s_inst, inst);
				for(unsigned wid = 0; wid < num_warps_per_shader; wid++) {
					SHADER_WARP_ID_Ty swid(shader_id, wid);
					printLiveInstStats(swid);
				}
				ptx_instruction* pinst = dynamic_cast<ptx_instruction*>(inst);
				if(pinst) {
					pinst->print_insn(stderr);
				} else {
					inst->print_insn(stderr);
				}
				assert(ilti);

			}
			stalled->mem_stalledby.insert(ilti->serial);
		}
	}

	if(!spfree) {
		for(itr = sp->begin(); itr != sp->end(); itr++) {
			warp_inst_t* inst = *itr;
			assert(inst != s_inst);
			InstLifeTimeInfo* ilti = findWarpInstAtPipelineStageEXAllWarps(shader_id, inst, PPL_AFTER_ISSUE);
			if(!ilti) {
				printf("insts %p blocked by %p on SP port.\n", s_inst, inst);
				ptx_instruction* pi = dynamic_cast<ptx_instruction*>(inst);
				s_inst->print_insn(stderr);
				fprintf(stderr, "\n");
				if(pi) {
					pi->print_insn(stderr);
				} else {
					inst->print_insn(stderr);
				}
				if(!ilti) {
					printLiveInstStats(swid);
					assert(ilti);
				}
			}
			stalled->sp_stalledby.insert(ilti->serial);
		}
	}

	if(!sfufree) {
		for(itr = sfu->begin(); itr != sfu->end(); itr++) {
			warp_inst_t* inst = *itr;
			assert(inst != s_inst);
			InstLifeTimeInfo* ilti = findWarpInstAtPipelineStageEXAllWarps(shader_id, inst, PPL_AFTER_ISSUE_TO_FU);
			if(!ilti) {
				ilti = findWarpInstAtPipelineStageEXAllWarps(shader_id, inst, PPL_AFTER_ISSUE);
			}
			if(!ilti) {

				for(unsigned wid = 0; wid < num_warps_per_shader; wid++) {
					SHADER_WARP_ID_Ty swid(shader_id, wid);
					printLiveInstStats(swid);
				}
				assert(ilti);
			}
			stalled->sfu_stalledby.insert(ilti->serial);
		}
	}
}

void Cartogarpher_clearAllLiveInstructionLog(shader_core_ctx* shader) {
	g_cartographer->clearAllLiveInstructionLog(shader);
}
void Cartographer::clearAllLiveInstructionLog(shader_core_ctx* shader) {
	unsigned num_warps = shader->m_config->max_warps_per_shader;
	unsigned shader_id = shader->get_sid();
	for(unsigned wid=0; wid<num_warps; wid++) {
		SHADER_WARP_ID_Ty swid = std::make_pair(shader_id, wid);
		if(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end()) {
			(m_live_inst_lifetimes[swid])->clear();
		}
	}
}

void Cartographer_clearLiveInstructionLog(shader_core_ctx* shader, unsigned warp_id) {
	g_cartographer->clearLiveInstructionLog(shader, warp_id);
}

void Cartographer::clearLiveInstructionLog(shader_core_ctx* shader, unsigned warp_id) {
	SHADER_WARP_ID_Ty swid = std::make_pair(shader->get_sid(), warp_id);
	if(m_live_inst_lifetimes.find(swid) != m_live_inst_lifetimes.end()) {
		printf("[Cartographer::clearLiveInstructionLog] Cleared (%d,%d)\n",
			swid.first, swid.second);
		for (std::list<InstLifeTimeInfo>::iterator itr = m_live_inst_lifetimes[swid]->begin();
				itr != m_live_inst_lifetimes[swid]->end(); itr++) {
			m_done_inst_lifetimes[swid]->push_back(*itr);
		}
		m_live_inst_lifetimes[swid]->clear();
	}
}

bool Cartographer_shouldLog(unsigned shader_id, unsigned warp_id) {
	return g_cartographer->shouldLog(shader_id, warp_id);
}
bool Cartographer::shouldLog(unsigned shader_id, unsigned warp_id) {
	if(num_warps_per_shader < warp_id + 1) { num_warps_per_shader = warp_id+1; }
	return shouldLog(shader_id, warp_id, gpu_tot_sim_cycle + gpu_sim_cycle);
}

void Cartographer_onDoneTMCommit(shader_core_ctx* core, unsigned warp_id) {
	g_cartographer->onDoneTMCommit(core, warp_id);
}
void Cartographer::onDoneTMCommit(shader_core_ctx* core, unsigned warp_id) {
	SHADER_WARP_ID_Ty swid = std::make_pair(core->get_sid(), warp_id);
	if (!shouldLog(core->get_sid(), warp_id)) return;

	assert (m_live_tcommit_serial.find(swid) != m_live_tcommit_serial.end());
	unsigned serial = m_live_tcommit_serial.at(swid);
	m_live_tcommit_serial.erase(swid);

	if (!linked_cartographerts or (!linked_cartographerts->account_committing_warps)) return;

	if (m_linked_commit_txns.find(serial) == m_linked_commit_txns.end()) {
		printf("[Cartographer::onDoneTMCommit] Looking for serial=%u in S%uW%u, but could not find.\n",
			serial, core->get_sid(), warp_id);
		return; // I need some logs. on 2015-09-02 // Error when executing <<<15,30>>> RBTree
	}
	std::set<MyCommittingTxnEntry*>* the_set = &(m_linked_commit_txns.at(serial));
	for (std::set<MyCommittingTxnEntry*>::iterator itr = the_set->begin(); itr != the_set->end(); itr++) {
		(*itr)->done_scoreboard_commit = getCurrentCycleCount();
	}
}

bool Cartographer::shouldLog(unsigned shader_id, unsigned warp_id, unsigned long long cycle) {
	if(!shaders_logged.empty()) {
		if(shaders_logged.find(shader_id)==shaders_logged.end()) return false;
	}
	// Cycle should always increment
	if(cycle_segments_logged.empty()) return true;
	if(curr_whitelist_cycle_seg_itr == cycle_segments_logged.end()) return false;
	const CYCLE_INTERVAL_Ty& curr_interval = *curr_whitelist_cycle_seg_itr;
	const unsigned long long lower = curr_interval.first, upper = curr_interval.second;
	if(cycle < lower) return false;
	else if(cycle > upper) {
		curr_whitelist_cycle_seg_itr++;
		return false;
	} else return true;
}

void updateInstSerailID(inst_t* inst, long sn) {
	g_cartographer->ldst_serials[inst] = sn;
}

void Cartographer_setShaderAndWarpID(unsigned shader_id, unsigned warp_id, unsigned dyn_warp_id) {
	g_cartographer->setShaderAndWarpID(shader_id, warp_id, dyn_warp_id);
}
void Cartographer_logCoalescingMemoryAccess(unsigned subwarp_id, new_addr_type addr, unsigned block, unsigned chunk) {
	g_cartographer->logCoalescingMemoryAccess(subwarp_id, addr, block, chunk);
}

void Cartographer::setShaderAndWarpID(unsigned shader_id, unsigned warp_id, unsigned dyn_warp_id) {
	if(!is_log_coalesced_addrs) return;
	coalesced_curr_swid = std::make_pair(shader_id, warp_id);
	coalesced_curr_dynamic_warpid = dyn_warp_id;
}
void Cartographer::logCoalescingMemoryAccess(unsigned subwarp_id, new_addr_type addr, unsigned block, unsigned chunk) {
	if(!is_log_coalesced_addrs) return;
	assert(coalesced_curr_swid.first != 0xFFFFFFFF);
	coalesced_accesses.push_back(CoalescedMemoryAccessInfo(
		coalesced_curr_swid, coalesced_curr_dynamic_warpid, subwarp_id, getCurrentCycleCount(), addr, block, chunk
	));
	if(coalesced_accesses.size() >= coalesced_db_write_interval) {
		dumpCurrentCoalescedToDB();
	}
}
void Cartographer::dumpCurrentCoalescedToDB() {
	clock_t tick, tock;
	tick = clock();
	unsigned long n_entries = coalesced_accesses.size();

	{
		int err;
		sqlite3* db;
		if((err = sqlite3_open("address_coalescing.db", &db)) != SQLITE_OK) {
			assert(0 && "SQLite database open error");
		} else {
			// Create table
			sqlite3_stmt* create_stmt;
			std::string create_query = "CREATE TABLE IF NOT EXISTS koalesced (CoreID INTEGER, WarpID INTEGER, DynamicWarpID INTEGER,";
			create_query += " SubwarpID INTEGER, Cycle INTEGER, Addr INTEGER, BlockAddr INTEGER, Chunk INTEGER";
			create_query += ");";
			sqlite3_prepare_v2(db, create_query.c_str(), (int)(create_query.size()),
				&create_stmt, NULL);
			assert((sqlite3_step(create_stmt) == SQLITE_DONE) && "Error creating table\n");
			sqlite3_finalize(create_stmt);
		}

		sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
		for(std::list<CoalescedMemoryAccessInfo>::iterator itr = coalesced_accesses.begin();
				itr != coalesced_accesses.end(); itr++) {
			CoalescedMemoryAccessInfo& cmai = *itr;
			cmai.insertToSQLiteDB(db);
		}
		sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
		printf("Inserted %lu entries to coalesced addresses list.\n", coalesced_accesses.size());
		sqlite3_close(db);
	}

	coalesced_accesses.clear();
	tock = clock();
	printf("[Cartographer] Cleared Coalesced Entries (%lu entries) and wrote to D.B. (%4f seconds)\n",
			n_entries, (double)(tock - tick)/CLOCKS_PER_SEC);
}

void CartographerTimeSeries_onPtxThreadStartTransaction(ptx_thread_info* thread) {
	g_cartographerts->onPtxThreadStartTransaction(thread);
}
void CartographerTimeSeries::onPtxThreadStartTransaction(ptx_thread_info* thread) {
	return;
	/*
	if (!log_events) return;
	struct TxnEventEntry ety = {};
	ety.event=MY_TXN_BEGIN;
	ety.ctaid=thread->m_ctaid;
	ety.tid=thread->m_tid,
	ety.num_transaction=thread->m_tm_num_transactions;
	ety.cycle = getCurrentCycleCount();
	events.push_back(ety);

	// Transaction Epoch Things
	CTAID_TID_Ty ctaidtid(ety.ctaid, ety.tid);

	struct MyTransactionThreadLifeTimeEntry* eety = new struct MyTransactionThreadLifeTimeEntry;
	eety->pc = thread->m_NPC; // see ptx_thread_info::tm_checkpoint
	eety->epoch = 0;
	if (m_done_txn_lifetime_entries.find(ctaidtid) == m_done_txn_lifetime_entries.end()) {
		m_done_txn_lifetime_entries[ctaidtid] = std::list<struct MyTransactionThreadLifeTimeEntry*>();
	}
	m_done_txn_lifetime_entries.at(ctaidtid).push_back(eety);

	assert (m_live_txn_lifetime_entries.find(ctaidtid) == m_live_txn_lifetime_entries.end());
	m_live_txn_lifetime_entries[ctaidtid] = eety;
	*/
}

void CartographerTimeSeries_onPtxThreadRollbackTransaction(ptx_thread_info* thread) {
	g_cartographerts->onPtxThreadRollbackTransaction(thread);
}
void CartographerTimeSeries::onPtxThreadRollbackTransaction(ptx_thread_info* thread) {
	return;
	/*
	if (!log_events) return;
	struct TxnEventEntry ety = {};
	ety.event=MY_TXN_ROLLBACK;
	ety.ctaid=thread->m_ctaid;
	ety.tid=thread->m_tid;
	ety.num_transaction=thread->m_tm_num_transactions;
	ety.cycle = getCurrentCycleCount();
	events.push_back(ety);

	// Transaction Epoch Things
	CTAID_TID_Ty ctaidtid(ety.ctaid, ety.tid);
	unsigned nextepoch = m_done_txn_lifetime_entries.at(ctaidtid).back()->epoch + 1;
	struct MyTransactionThreadLifeTimeEntry* eety = new struct MyTransactionThreadLifeTimeEntry;
	eety->pc = thread->m_NPC; // see ptx_thread_info::tm_checkpoint
	eety->epoch = nextepoch;
	m_done_txn_lifetime_entries.at(ctaidtid).push_back(eety);

	assert (m_live_txn_lifetime_entries.find(ctaidtid) != m_live_txn_lifetime_entries.end());
	m_live_txn_lifetime_entries[ctaidtid] = eety;
	*/
}

void CartographerTimeSeries_onPtxThreadCommitTransaction(ptx_thread_info* thread) {
	g_cartographerts->onPtxThreadCommitTransaction(thread);
}
void CartographerTimeSeries::onPtxThreadCommitTransaction(ptx_thread_info* thread) {
	return;
	/*
	if (!log_events) return;
	struct TxnEventEntry ety = {};
	ety.event=MY_TXN_COMMIT;
	ety.ctaid=thread->m_ctaid;
	ety.tid=thread->m_tid;
	ety.num_transaction=thread->m_tm_num_transactions;
	ety.cycle = getCurrentCycleCount();
	events.push_back(ety);

	CTAID_TID_Ty ctaidtid(ety.ctaid, ety.tid);
	assert (m_live_txn_lifetime_entries.find(ctaidtid) != m_live_txn_lifetime_entries.end());
	m_live_txn_lifetime_entries.erase(ctaidtid);
	*/
}

void MyThreadEventParser::finalize() {
	if (curr_event != NULL) do_advanceEvent(curr_event);
}

void MyThreadEventParser::handleCurrEvent(MyThreadEvent* currevt) {
	if (curr_event != NULL) {
		do_advanceEvent(curr_event);
		if (prev_event != NULL) delete prev_event;
		prev_event = curr_event;
		curr_event = currevt;
	}  else {
		curr_event = currevt;
		prev_event = NULL;
	}
}

void MyThreadEventParser::singleStepEvent(MyThreadEvent* evt) {
	do_advanceEvent(evt);
	prev_event = evt;
}

void MyThreadEventParser::do_advanceEvent(MyThreadEvent* evt) {
	if (evt != NULL) {
		unsigned tmp = 0;
		if (prev_event == NULL) { tmp = 0; }
		else {
			tmp = evt->cycle - prev_event->cycle;
		}
		const unsigned delta_cycle = tmp;

		if (dynamic_cast<MyThreadInitEvent*>(evt)) {
			if (state == EXITED) {
				state = RUNNING;
				cycles += delta_cycle;
				if (!is_active) cycles_inactive += delta_cycle;
				prev_exec_begin = evt->cycle;
			} else {
				is_parse_error = 1;
			}
		} else if (dynamic_cast<MyThreadExitEvent*>(evt)) {
			if (state == RUNNING) {
				state = EXITED;
				num_init ++;
				cycles += delta_cycle;
				if (!is_active) cycles_inactive += delta_cycle;
			} else {
				is_parse_error = 2;
			}
		} else if (dynamic_cast<MyThreadStartScoreboardCommitEvent*>(evt)) {
			if (state == RUNNING) {
				state = IN_SCOREBOARD_COMMIT;
				curr_tx_pass = -999;
				prev_commit_begin = evt->cycle; cycles += delta_cycle;
				if (!is_active) cycles_inactive += delta_cycle;
			} else {
				is_parse_error = 3;
				printf("Expecting state %d, got %d\n", (int)RUNNING, (int)state);
			}
		} else if (dynamic_cast<MyThreadDoneScoreboardCommitEvent*>(evt)) {
			if (state == IN_SCOREBOARD_COMMIT || state == IN_TLW_CU_COMMUNICATION) {
				state = RUNNING;
				num_scoreboard_commit ++;
				num_tx ++;
				if (curr_tx_pass == -999) {
					is_parse_error = 14;
					printf("Could not successfully determine the state of the current txn.\n");
				} else {
					if (prev_commit_begin == -999) {
						is_parse_error = 17;
						printf("The cycle at which the previous commit began is not known at ScbCommit, ERROR!!!\n");
					}
					if (curr_tx_pass == 0) {
						num_cu_aborts ++;
						commit_cycle_fail += (evt->cycle - prev_commit_begin);
					} else if (curr_tx_pass == 1) {
						num_commits ++;
						commit_cycle_pass += (evt->cycle - prev_commit_begin);
					}
				}
				cycles += delta_cycle;
				if (!is_active) { cycles_inactive += delta_cycle; }

				curr_tx_pass = -999;
				prev_commit_begin = -999;
			} else {
				is_parse_error = 4;
			}
		} else if (dynamic_cast<MyThreadFailPreCommitValidation*>(evt)) {
			if (state == IN_SCOREBOARD_COMMIT) {
				state = RUNNING; num_pre_cu_abort ++;
				curr_tx_pass = 0;
				cycles += delta_cycle;
				if (!is_active) cycles_inactive += delta_cycle;
				else {
					if (prev_commit_begin == -999) {
						is_parse_error = 20;
						printf("The cycle at which the previous commit began is not known at PCVFail, ERROR!!!\n");
					}
					commit_cycle_fail += evt->cycle - prev_commit_begin;
					prev_commit_begin = -999;
				}
			} else {
				is_parse_error = 5;
			}
		} else if (dynamic_cast<MyThreadDoneCoreSideReadonlyCommitEvent*>(evt)) {
			// I don't exactly know what's happening here
			if (state == IN_SCOREBOARD_COMMIT) {
				cycles += delta_cycle;
				curr_tx_pass = 1;
			} else {
				is_parse_error = 21;
				printf("Encountered read-only core-side commit when current");
			}
		} else if (dynamic_cast<MyThreadScoreboardHazardEvent*>(evt)) {
			if (is_active == 0) {
				is_parse_error = 16;
				printf("Only an active thread can encounter an SB hazard.\n");
			}
			if (state == RUNNING) {
				MyThreadScoreboardHazardEvent* she = dynamic_cast<MyThreadScoreboardHazardEvent*>(evt);
				cycles += delta_cycle; // Must be active, do not increment cycles_inactive
				std::string key = she->reason;
				if (cycle_hazards.find(key) == cycle_hazards.end()) cycle_hazards[key] = 0;
				cycle_hazards[key] += she->until - she->cycle;
			} else {
				is_parse_error = 6;
				printf("Expecting state=%d, got %d @ %llu\n", RUNNING, state, evt->cycle);
			}
		} else if (dynamic_cast<MyThreadTLWStateChangedEvent*>(evt)) {
			MyThreadTLWStateChangedEvent* ttsce = dynamic_cast<MyThreadTLWStateChangedEvent*>(evt);
			tx_log_walker::commit_tx_state_t cs = ttsce->state;
			num_tlw_state_change++;
			cycles += delta_cycle;
			if (!is_active) cycles_inactive += delta_cycle;
			if (state == IN_SCOREBOARD_COMMIT) {
				if (cs != tx_log_walker::INTRA_WARP_CD && cs != tx_log_walker::ACQ_CID) {
					is_parse_error = 7;
					printf("Expecting commit_tx_state_t %d or %d, got %d\n",
						tx_log_walker::INTRA_WARP_CD, tx_log_walker::ACQ_CID,
						cs);
				} else {
					commit_tx_state = cs;
					state = IN_TLW_CU_COMMUNICATION;
				}
			} else if (state == IN_TLW_CU_COMMUNICATION) {
				switch (cs) {
					case tx_log_walker::SEND_RS : {
						if (commit_tx_state == tx_log_walker::INTRA_WARP_CD ||
							commit_tx_state == tx_log_walker::ACQ_CID) {
							commit_tx_state = cs;
						} else {
							is_parse_error = 9;
							printf("Expecting previous commit_tx_state_t %d or %d, got %d\n",
								tx_log_walker::INTRA_WARP_CD, tx_log_walker::ACQ_CID, commit_tx_state);
						}
						break;
					}
					case tx_log_walker::SEND_WS: {
						if (commit_tx_state == tx_log_walker::SEND_RS) {
							commit_tx_state = cs;
						} else {
							is_parse_error = 10;
							printf("Expecting previous commit_tx_state_t %d, got %d\n",
								tx_log_walker::SEND_RS, commit_tx_state);
						}
						break;
					}
					case tx_log_walker::WAIT_CU_REPLY: {
						if (commit_tx_state == tx_log_walker::SEND_WS) {
							commit_tx_state = cs;
						} else {
							is_parse_error = 11;
							printf("Expecting previous commit_tx_state_t %d, got %d\n",
								tx_log_walker::SEND_WS, commit_tx_state);
						}
						break;
					}
					case tx_log_walker::SEND_ACK_CLEANUP: {
						if (commit_tx_state == tx_log_walker::WAIT_CU_REPLY) {
							commit_tx_state = cs;
							if (ttsce->is_passed == 1) curr_tx_pass = 1;
							else curr_tx_pass = 0;
						} else {
							is_parse_error = 12;
							printf("Expecting previous commit_tx_state_t %d, got %d\n",
								tx_log_walker::WAIT_CU_REPLY, commit_tx_state);
						}
						break;
					}
					case tx_log_walker::IDLE: {
						if (commit_tx_state == tx_log_walker::SEND_ACK_CLEANUP) {
							commit_tx_state = cs;
						} else {
							is_parse_error = 13;
							printf("Expecting previous commit_tx_state_t %d, got %d\n",
								tx_log_walker::SEND_ACK_CLEANUP, commit_tx_state);
						}
						break;
					}
					case tx_log_walker::INTRA_WARP_CD:
					case tx_log_walker::ACQ_CID:
						// Handled in the above if statement already
						break;
					case tx_log_walker::ACQ_CU_ENTRIES:
					case tx_log_walker::WAIT_CU_ALLOC_REPLY:
					case tx_log_walker::RESEND_RS:
					case tx_log_walker::RESEND_WS:
						break; // These are not used in warptm therefore do not need handling
				}
			} else {
				is_parse_error = 8;
				printf("Expecting state %d, got %d\n", IN_SCOREBOARD_COMMIT, state);
			}
		} else if (dynamic_cast<MyThreadActiveMaskChangedEvent*>(evt)) {
			MyThreadActiveMaskChangedEvent* tamce = dynamic_cast<MyThreadActiveMaskChangedEvent*>(evt);
			cycles += delta_cycle;
			if (!is_active) cycles_inactive += delta_cycle;
			if (is_active != tamce->changed_to) {
				num_active_mask_change ++;
			}

			if (tamce->changed_to == 0) {
				if (dynamic_cast<MyThreadDoneScoreboardCommitEvent*>(prev_event)) {
					prev_donecommit_inactive = evt->cycle;
				} else if (dynamic_cast <MyThreadTLWStateChangedEvent*>(prev_event) and
					tamce->reason == "TXRestart") { // Should be PASS
					prev_donecommit_inactive = evt->cycle;
				}
			} else if (tamce->changed_to == 1) {
				if (prev_donecommit_inactive != -999) {
					inactive_cycle_incommit += delta_cycle;
					prev_donecommit_inactive = -999;
				}
			}

			is_active = tamce->changed_to;
		}
	}
}

long MyThreadEventParser::getTotalHazardCycles() {
	long ret = 0;
	for (std::map<std::string, long>::iterator itr = cycle_hazards.begin();
		itr != cycle_hazards.end(); itr++) {
		ret += itr->second;
	}
	return ret;
}
void MyThreadEventParser::printHeader(FILE* f) {
	printf("Divergence, WaitOtherTxns, CommitPass, CommitFail, "
			"Exec");
	for (std::map<std::string, long>::iterator itr = cycle_hazards.begin();
		itr != cycle_hazards.end(); itr++) {
		printf(", %s", itr->first.c_str());
	}
	printf("\n");
}

// Multiplier needed for printing average
void MyThreadEventParser::print(FILE* f, double multiplier) {
	printf("%.2f, ", (cycles_inactive - inactive_cycle_incommit) * multiplier);
	printf("%.2f, ", inactive_cycle_incommit * multiplier);
	printf("%.2f, ", commit_cycle_pass * multiplier);
	printf("%.2f, ", commit_cycle_fail * multiplier);
	printf("%.2f", computeExecCycle() * multiplier);
	for (std::map<std::string, long>::iterator itr = cycle_hazards.begin();
			itr != cycle_hazards.end(); itr++) {
			printf(", %.2f", itr->second * multiplier);
		}
	printf("\n");
}

void CartographerTimeSeries::summary() {
	unsigned long instcnt = 0;
	{
		std::list<unsigned long>::iterator   itr1 = num_insts_per_interval.begin();
		for (; itr1 != num_insts_per_interval.end(); itr1++) instcnt += (*itr1);
	}
	printf("[CartographerTimeSeries] L1 Stats\nL1 Access = %lu\n", l1_stats.l1_access_count);
	printf("  - L1D   All Access = %lu, RF = %lu\n", l1_stats.l1d_access_count, l1_stats.l1d_rf_count);
	printf("    Where Txn Access = %lu, RF = %lu\n", l1_stats.l1d_access_count_txn, l1_stats.l1d_rf_count_txn);
	printf("  - L1C   All Access = %lu, RF = %lu\n", l1_stats.l1c_access_count, l1_stats.l1c_rf_count);
	printf("  - L1I   All Access = %lu, RF = %lu\n", l1_stats.l1i_access_count, l1_stats.l1i_rf_count);
	printf("  - L1T   All Access = %lu, RF = %lu\n", l1_stats.l1t_access_count, l1_stats.l1t_rf_count);
	printf("[CartographerTimeSeries] Inst count = %lu\n", instcnt);

	printf("----------------[ Per interval states ]-----------------\n");
	printf("TotAcc\tL1DA\tL1DRF\tL1DATxn\tL1DRFTxn\tL1CA\tL1CRF\tL1IA\tL1IRF\tL1TA\tL1TRF\tInstCompleted\n");

	std::list<struct MyL1Stats>::iterator itr = l1_per_interval.begin();
	std::list<unsigned long>::iterator   itr1 = num_insts_per_interval.begin();
	for (; itr != l1_per_interval.end(); itr++, itr1++) {
		struct MyL1Stats* s = &(*itr);
		printf("%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\t%lu\n",
			s->l1_access_count, s->l1d_access_count, s->l1d_rf_count, s->l1d_access_count_txn, s->l1d_rf_count_txn,
			s->l1c_access_count,s->l1c_rf_count,     s->l1i_access_count,s->l1i_rf_count,
			s->l1t_access_count, s->l1t_rf_count,    *itr1);
	}

	if (verified) { printf("[CartographerTimeSeries] Cache Stat result check OK.\n"); }
	else { printf("[CartographerTimeSeries] Cache Stat result check NOT OK.\n"); }

	printf("Maximum CID in Flight: %u\n", max_cid_in_flight);
	printf("CIDs linked with InstructionLifeTime: %u\n", num_linked_with_sandwich);
	if (txn_cid_in_flight.size() == 0) {
		printf("[CartograperTimeSeries] In-flight Commit ID all clear at program exit.\n");
	} else {
		printf("[CartograperTimeSeries] In-flight Commit NOT all clean (We have %lu remaining at exit.)\n",
			txn_cid_in_flight.size());
	}

	if (m_live_txn_lifetime_entries.size() == 0) {
		printf("[CartographerTimeSeries] In-flight Transactions all clear at program exit.\n");
	} else {
		printf("[CartographerTimeSeries] In-flight Transactions not all clear (%lu remaining at exit.)\n",
			m_live_txn_lifetime_entries.size());
	}

	// Epoch Count
	{
		unsigned epoch_count = 0;
		for (std::map<CTAID_TID_Ty, std::list<MyTransactionThreadLifeTimeEntry*>>::iterator itr = m_done_txn_lifetime_entries.begin();
				itr != m_done_txn_lifetime_entries.end(); itr++) {
			epoch_count += itr->second.size();
		}
		printf("[CartographerTimeSeries] Total Transaction Epoch Count = %u\n",epoch_count);
	}

	std::map<CTAID_TID_Ty, std::map<std::string, unsigned> > thd_state_hist_1; // To compare against allcycles
	bool is_all_ok = true;

	// Check Thread Event List
	{
		MyThreadEventParser sum_parser; unsigned num_threads = 0;
		long max_cycles = 0, sum_cycles = 0;
		printf("Parsing mode: %d\n", parsing_mode);

		std::set<CTAID_TID_Ty> ctaidtids;
		if (parsing_mode == 1) {
			if (is_always_print_all_events) {
				printf("[All events list]\n");
			}
			std::unordered_map<CTAID_TID_Ty, std::deque<MyThreadEvent*>>::iterator itr = thd_events.begin();
			for (; itr != thd_events.end(); itr++) {
				ctaidtids.insert (itr->first);
				if (is_always_print_all_events) {
					CTAID_TID_Ty x = itr->first;
					printf("(%u,%u,%u)-(%u,%u,%u) ",
						x.ctaid.x, x.ctaid.y, x.ctaid.z, x.tid.x, x.tid.y, x.tid.z);
					std::deque<MyThreadEvent*>* lst = &(itr->second);
					for (std::deque<MyThreadEvent*>::iterator itr2 = lst->begin(); itr2!=lst->end(); itr2++) {
						(*itr2)->print(stdout);
					}
					printf("\n");
				}
			}
		} else {
			std::unordered_map<CTAID_TID_Ty, MyThreadEventParser*>::iterator itr = thd_event_parsers.begin();
			for (; itr != thd_event_parsers.end(); itr++) {
				ctaidtids.insert (itr->first);
			}
		}

		std::set<CTAID_TID_Ty>::iterator itr = ctaidtids.begin();
		for (; itr != ctaidtids.end(); itr++) {
			CTAID_TID_Ty ctaidtid = *itr;
			num_threads++;
			MyThreadEventParser* parser = NULL;
			if (parsing_mode == 1) { // Store and then parse
				std::deque<MyThreadEvent*>* evtlist = &(thd_events.at(*itr));
				parser = new MyThreadEventParser();
				for (std::deque<MyThreadEvent*>::iterator itr1 = evtlist->begin(); itr1!=evtlist->end(); itr1++) {
					parser->singleStepEvent(*itr1);
				}
			} else if (parsing_mode == 2) {
				parser = (thd_event_parsers.at(*itr));
				parser->finalize();
			}
			if (parser->is_parse_error == 0) {
				sum_parser.add(*parser);
				if (max_cycles < parser->cycles) max_cycles = parser->cycles;
				sum_cycles += parser->cycles;
				thd_state_hist_1[ctaidtid] = std::map<std::string, unsigned>();
				thd_state_hist_1[ctaidtid]["active"] = parser->cycles - parser->cycles_inactive;
				thd_state_hist_1[ctaidtid]["inactive"] = parser->cycles_inactive;
			} else {
				printf("is_parse_error = %d\n", parser->is_parse_error);
				is_all_ok = false;
			}

			if (parsing_mode == 1) delete parser;
		}

		if(is_all_ok) {
			printf("[CartographerTimeSeries] All thread events parse ok.\n");
		} else {
			printf("[CartographerTimeSeries] Not all thread events ok.\n");
		}
		printf("Number of init-exit pairs: %u\n", sum_parser.num_init);
		printf("Number of scoreboard commit start-done pairs: %u\n", sum_parser.num_commits);
		printf("Number of pre-commit validation abort: %u\n", sum_parser.num_pre_cu_abort);
		printf("Total cycles: %ld, cycles-in-hazard: %ld\n", sum_parser.cycles, sum_parser.getTotalHazardCycles());

		printf("Average/Max cycles across %u threads: %.2f/%ld\n", num_threads, sum_cycles*1.0/num_threads, max_cycles);

		printf("Total TLW state change: %u\n", sum_parser.num_tlw_state_change);
		printf("Transactions = %u = %u Commits + %u CU aborts + %u pre-CU aborts\n",
			sum_parser.num_cu_aborts + sum_parser.num_pre_cu_abort + sum_parser.num_commits,
			sum_parser.num_commits,
			sum_parser.num_cu_aborts,
			sum_parser.num_pre_cu_abort);
		printf("Total active mask changes: %u\n", sum_parser.num_active_mask_change);

		printf("---------------8<-----------------\n");
		sum_parser.printHeader(stdout);
		sum_parser.print(stdout, 1.0f / num_threads);
		printf("--------------->8-----------------\n");
	}


	{
		if (sid_wid_to_threads.empty()) {
			printf("[CartographerTimeSeries] SID-WID to Threads map clear.\n");
		} else printf("[CartographerTimeSeries] SID-WID to Threads map not clear: %lu entries left",
			sid_wid_to_threads.size());
	}

	{
		dumpThreadStateHist();
	}

	if (account_every_cycle && log_events) {
		unsigned num_check_ok = 0, num_all = 0, sum_abs_diff = 0;
		printf("Thread cycle breakdown\n");
		std::map<CTAID_TID_Ty, std::map<std::string, unsigned> >::iterator itr = thd_cycle_breakdown_ref.begin();
		for (; itr != thd_cycle_breakdown_ref.end(); itr++) {
			num_all ++;
			CTAID_TID_Ty ctaidtid = itr->first;
			std::map<std::string, unsigned>* bd = &(itr->second);
			printf("(%u,%u,%u)-(%u,%u,%u):", ctaidtid.ctaid.x, ctaidtid.ctaid.y, ctaidtid.ctaid.z,
				ctaidtid.tid.x, ctaidtid.tid.y, ctaidtid.tid.z);
			std::map<std::string, unsigned>* to_check;
			if (is_all_ok) {
				assert(thd_state_hist_1.find(ctaidtid) != thd_state_hist_1.end());
				to_check = &(thd_state_hist_1.at(ctaidtid));
			}
			bool is_all_check_pass = true;
			for (std::map<std::string, unsigned>::iterator itr1 = bd->begin(); itr1 != bd->end(); itr1++) {
				printf(" %s:%u", itr1->first.c_str(), itr1->second);
				int diff = to_check->at(itr1->first) - itr1->second;
				if (itr1->first == "inactive") { if(diff!=0) is_all_check_pass = false; }
				else { if (diff > 50 or diff < -50) is_all_check_pass = false; }
				sum_abs_diff += abs(diff);
			}

			if (is_all_check_pass) { printf(" Check OK"); num_check_ok ++; }
			else {
				printf(" Not OK... ");
				if (thd_state_change_history.find(ctaidtid) != thd_state_change_history.end()) {
					for (std::list<struct MyThreadStateChangeTy>::iterator itr =
						thd_state_change_history[ctaidtid].begin(); itr !=
								thd_state_change_history[ctaidtid].end(); itr++) {
						printf(" [%d->%d @ %llu]", itr->state0, itr->state1, itr->cycle);
					}
				}
			}
			printf("\n");
		}
		if (is_all_ok) {
			printf("[CartographerTimeSeries] Parsed vs brute-force thd state distrib: OK=%u/%u, sum/avg absdiff=%u/%.2f\n",
				num_check_ok, num_all, sum_abs_diff, sum_abs_diff*1.0/num_all);
		}
	}
}


void CartographerTimeSeries::do_incrementL1AccessCount(const char* which, int status, struct MyL1Stats* stats) {
	bool is_rf = false;

		if (status == (int)RESERVATION_FAIL) is_rf = true;
		if (!is_rf) stats->l1_access_count ++;
		else stats->l1_rf_count ++;

		if (!strcmp(which, "L1D") || !strcmp(which, "L1D_Txn")) {
			if (!is_rf) stats->l1d_access_count ++;
			else stats->l1d_rf_count ++;
			if (!strcmp(which, "L1D_Txn")) {
				if (!is_rf) stats->l1d_access_count_txn ++;
				else stats->l1d_rf_count_txn ++;
			}
		}
		else if(!strcmp(which, "L1T")) {
			if (!is_rf) stats->l1t_access_count ++;
			else stats->l1t_rf_count ++;
		}
		else if(!strcmp(which, "L1C")) {
			if (!is_rf) stats->l1c_access_count ++;
			else stats->l1c_rf_count ++;
		}
		else if(!strcmp(which, "L1I")) {
			if (!is_rf) stats->l1i_access_count ++;
			else stats->l1i_rf_count ++;
		}
		else assert(0);
}

void CartographerTimeSeries_incrementL1AccessCount(const char* which, enum cache_request_status status) {
	g_cartographerts->incrementL1AccessCount(which, (int)status);
}
void CartographerTimeSeries::incrementL1AccessCount(const char* which, int status) {
	do_incrementL1AccessCount(which, status, &(this->l1_stats));
	// Interval stuff
	unsigned long long cycle = getCurrentCycleCount();
	if (cycle >= curr_l1_interval_begin) {
		curr_l1_interval_begin += interval;
		struct MyL1Stats stats = {};
		l1_per_interval.push_back(stats);
	}
	do_incrementL1AccessCount(which, status, &(l1_per_interval.back()));
}

MyThreadEvent* CartographerTimeSeries::getCurrEvent(const CTAID_TID_Ty& the_id) {
	if (parsing_mode == 1) {
		if (thd_events.find(the_id) == thd_events.end()) return NULL;
		else return thd_events.at(the_id).back();
	} else if (parsing_mode == 2) {
		if (thd_event_parsers.find(the_id) == thd_event_parsers.end()) return NULL;
		else return thd_event_parsers.at(the_id)->getCurrentEvent();
	} else assert(0);
}

void CartographerTimeSeries::appendEvent (const CTAID_TID_Ty& the_id, MyThreadEvent* evt) {
	if (parsing_mode == 1) {
		if (thd_events.find(the_id) == thd_events.end())
			thd_events[the_id] = std::deque<MyThreadEvent*>();
		thd_events.at(the_id).push_back(evt);
	} else if(parsing_mode==2) {
		if (thd_event_parsers.find(the_id) == thd_event_parsers.end())
			thd_event_parsers[the_id] = new MyThreadEventParser();
		MyThreadEventParser* parser = thd_event_parsers.at(the_id);
		parser->handleCurrEvent(evt);
	}
}

struct MyL1Stats;

static bool PrintCheckEquals(unsigned a, unsigned b, const char* desc) {
	if (a==b) { printf("[CartoTS] %s OK: %u\n", desc, a); return true; }
	else { printf("[CartoTS] %s NOT OK: %u vs %u\n", desc, a, b); return false; }
}

const char* MyThreadTLWStateChangedEvent::ctsz[] = {
	"IDLE",
	"INTRA_WARP_CD",
	"ACQ_CID",
	"ACQ_CU_ENTRIES",
	"WAIT_CU_ALLOC_REPLY",
	"SEND_RS",
	"SEND_WS",
	"WAIT_CU_REPLY",
	"RESEND_RS",
	"RESEND_WS",
	"SEND_ACK_CLEANUP"
};

void CartographerTimeSeries_checkL1Accesses(unsigned l1_tot, unsigned l1c, unsigned l1i, unsigned l1t, unsigned l1d) {
	g_cartographerts->checkL1Accesses(l1_tot, l1c, l1i, l1t, l1d);
}
void CartographerTimeSeries::checkL1Accesses(unsigned l1_tot, unsigned l1c, unsigned l1i, unsigned l1t, unsigned l1d) {
	if (!account_every_cycle) return;
	verified &= PrintCheckEquals(l1_tot, this->l1_stats.l1_access_count, "L1 Total Access Count");
	verified &= PrintCheckEquals(l1c,    this->l1_stats.l1c_access_count,"L1C Total Access Count");
	verified &= PrintCheckEquals(l1i,    this->l1_stats.l1i_access_count,"L1I Total Access Count");
	verified &= PrintCheckEquals(l1t,    this->l1_stats.l1t_access_count,"L1T Total Access Count");
	verified &= PrintCheckEquals(l1d,    this->l1_stats.l1d_access_count,"L1D Total Access Count");
}

void CartographerTimeSeries_checkL1ReservationFails(unsigned l1_rf, unsigned l1c_rf, unsigned l1i_rf, unsigned l1t_rf, unsigned l1d_rf) {
	g_cartographerts->checkL1ReservationFails(l1_rf, l1c_rf, l1i_rf, l1t_rf, l1d_rf);
}
void CartographerTimeSeries::checkL1ReservationFails(unsigned l1_rf, unsigned l1c_rf, unsigned l1i_rf, unsigned l1t_rf, unsigned l1d_rf) {
	if (!account_every_cycle) return;
	verified &= PrintCheckEquals(l1_rf, this->l1_stats.l1_rf_count, "L1 Total Reservation Fail Count");
	verified &= PrintCheckEquals(l1c_rf, this->l1_stats.l1c_rf_count, "L1C Total Reservation Fail Count");
	verified &= PrintCheckEquals(l1i_rf, this->l1_stats.l1i_rf_count, "L1I Total Reservation Fail Count");
	verified &= PrintCheckEquals(l1t_rf, this->l1_stats.l1t_rf_count, "L1T Total Reservation Fail Count");
	verified &= PrintCheckEquals(l1d_rf, this->l1_stats.l1d_rf_count, "L1D Total Reservation Fail Count");
}

void CartographerTimeSeries_incrementInstCount(unsigned count) {
	g_cartographerts->incrementInstCount(count);
}
void CartographerTimeSeries::incrementInstCount(unsigned count) {
	if (!account_every_cycle) return;
	num_insts_completed += count;
	unsigned long long cycle = getCurrentCycleCount();
	if (cycle >= curr_insts_interval_begin) {
		curr_insts_interval_begin += interval;
		num_insts_per_interval.push_back(0);
	}
	num_insts_per_interval.back() += count;
}

MyCommittingTxnEntry* CartographerTimeSeries::locateMyCommittingTxnEntryByCID(unsigned commit_id) {
	struct MyCommittingTxnEntry* ret = NULL;
	unsigned num_found = 0;
	for (std::vector<MyCommittingTxnEntry*>::iterator itr = txn_cid_in_flight.begin();
		itr != txn_cid_in_flight.end(); itr++) {
		if ((*itr)->cid == commit_id) {
			ret = (*itr);
			num_found ++;
		}
	}
	assert (num_found <= 1);
	return ret;
}

// 2015-07-14: Move the tracking statements from the functional simulation side to
//             the timing simulation side.
void CartographerTimeSeries_onTxnAcquiredCID(shader_core_ctx* core, unsigned wid,
		struct tx_log_walker::warp_commit_tx_t* commit_warp_info, struct tx_log_walker::commit_tx_t* tx,
		int tid_in_warp) {
	g_cartographerts->onTxnAcquiredCID(core, wid, commit_warp_info, tx, tid_in_warp);
}
void CartographerTimeSeries::onTxnAcquiredCID(shader_core_ctx* core, unsigned wid,
		struct tx_log_walker::warp_commit_tx_t* commit_warp_info, struct tx_log_walker::commit_tx_t* tx,
		int tid_in_warp) {
	if (!account_committing_warps) return;

	if (dump_txn_rwsets) {
		unsigned long long cycle = getCurrentCycleCount();

		tm_manager* tmm = dynamic_cast<tm_manager*> (tx->m_tm_manager);
		assert (tmm);

		const char enter_exit = 1;

		gzwrite(f_rwsets, &cycle, sizeof(unsigned long long));
		gzwrite(f_rwsets, &enter_exit, sizeof(char));

		dim3 ctaid3 = tx->m_tm_manager->m_thread->get_ctaid();
		dim3 tid3   = tx->m_tm_manager->m_thread->get_tid();

		gzwrite(f_rwsets, &ctaid3, sizeof(dim3));
		gzwrite(f_rwsets, &tid3,   sizeof(dim3));

		char rw = 'R'; unsigned sz = tmm->m_read_set.size();
		gzwrite(f_rwsets, &rw, sizeof(char));
		gzwrite(f_rwsets, &sz, sizeof(unsigned));

		addr_set_t::iterator itr = tmm->m_read_set.begin();
		for (; itr != tmm->m_read_set.end(); itr++) {
			unsigned addr = *itr;
			gzwrite(f_rwsets, &addr, sizeof(unsigned));
		}

		rw = 'W'; sz = tmm->m_write_set.size();
		gzwrite(f_rwsets, &rw, sizeof(char));
		gzwrite(f_rwsets, &sz, sizeof(unsigned));

		itr = tmm->m_write_set.begin();
		for (; itr != tmm->m_write_set.end(); itr++) {
			unsigned addr = *itr;
			gzwrite(f_rwsets, &addr, sizeof(unsigned));
		}

		unsigned cid = tx->m_commit_id;
		read_sets[cid] = addr_set_t();
		read_sets[cid].insert(tmm->m_read_set.begin(), tmm->m_read_set.end());
		write_sets[cid] = addr_set_t();
		write_sets[cid].insert(tmm->m_write_set.begin(), tmm->m_write_set.end());
	}

	MyCommittingTxnEntry* ety = new MyCommittingTxnEntry(core);
	ety->cid = tx->m_commit_id; ety->wid = wid;
	ety->tid_in_warp = tid_in_warp;
	ety->commit_warp_info = *commit_warp_info;
	ety->has_done_fill = false;
	ety->send_rs_started = getCurrentCycleCount();

	// TODO: Reenable this
	if (g_cartographer && (getenv("TOMMY_FLAG_0713")==NULL)) {
		SHADER_WARP_ID_Ty swid (core->m_sid, wid);
		g_cartographer->linkCommittingTxnToWarp(swid, ety); // Takes care of SN
		assert(core->m_scoreboard->m_in_tx_commit.at(wid) == 1);
		num_linked_with_sandwich ++;
	}

	this->txn_cid_in_flight.push_back(ety);
	if (max_cid_in_flight < ety->cid) max_cid_in_flight = ety->cid;
	core->m_thread[wid * core->m_warp_size + tid_in_warp]->m_is_committing = true;
}

void CartographerTimeSeries_onTxnStartedSendingWS(unsigned cid) {
	g_cartographerts->onTxnStartSendingWS(cid);
}
void CartographerTimeSeries::onTxnStartSendingWS(unsigned cid) {
	if (!account_committing_warps) return;
	MyCommittingTxnEntry* mcte = locateMyCommittingTxnEntryByCID(cid);
	assert (mcte);
	mcte->send_ws_started = getCurrentCycleCount();
}

void CartographerTimeSeries::do_onTxnSendOneRWEntry(unsigned cid, char rw) {
	if (!account_committing_warps) return;
	MyCommittingTxnEntry* mcte = locateMyCommittingTxnEntryByCID(cid);
	assert (mcte);
	if (rw == 'R') mcte->rs_etys_time.push_back(getCurrentCycleCount());
	else if (rw == 'W') mcte->ws_etys_time.push_back(getCurrentCycleCount());
	else assert(0 and "rw must be R or W");
}

void CartographerTimeSeries_onTxnSendOneRSEntry(unsigned cid) {g_cartographerts->onTxnSendOneRSEntry(cid);}
void CartographerTimeSeries::onTxnSendOneRSEntry(unsigned cid) {
	if (account_committing_warps) do_onTxnSendOneRWEntry(cid, 'R');
}
void CartographerTimeSeries_onTxnSendOneWSEntry(unsigned cid) {g_cartographerts->onTxnSendOneWSEntry(cid);}
void CartographerTimeSeries::onTxnSendOneWSEntry(unsigned cid) {
	if (account_committing_warps) do_onTxnSendOneRWEntry(cid, 'W');
}

void CartographerTimeSeries_onTxnDoneFill(unsigned cid) {g_cartographerts->onTxnDoneFill(cid);}
void CartographerTimeSeries::onTxnDoneFill(unsigned cid) {
	if (!account_committing_warps) return;
	struct MyCommittingTxnEntry* ety = locateMyCommittingTxnEntryByCID(cid);
	assert (ety != NULL);
	ety->has_done_fill = true;
	ety->done_fill = getCurrentCycleCount();
}
void CartographerTimeSeries_onTxnReceivedCUReply(unsigned cid) {g_cartographerts->onTxnReceivedCUReply(cid);}
void CartographerTimeSeries::onTxnReceivedCUReply(unsigned cid) {
	if (!account_committing_warps) return;
	struct MyCommittingTxnEntry* ety = locateMyCommittingTxnEntryByCID(cid);
	assert (ety != NULL); ety->cu_replies_time.push_back(getCurrentCycleCount());
}
void CartographerTimeSeries_onTxnSendTXPassFail(unsigned cid) {g_cartographerts->onTxnSendTXPassFail(cid);}
void CartographerTimeSeries::onTxnSendTXPassFail(unsigned cid) {
	if (!account_committing_warps) return;
	struct MyCommittingTxnEntry* ety = locateMyCommittingTxnEntryByCID(cid);
	assert (ety != NULL); ety->send_tx_passfail = getCurrentCycleCount();
}

// Main purpose is to set m_is_committing to false.
// Should also be set false through abort()
void CartographerTimeSeries_onTxnReceivedAllCUReplies(shader_core_ctx* core, unsigned wid,
		struct tx_log_walker::warp_commit_tx_t* commit_warp_info, struct tx_log_walker::commit_tx_t* tx,
		const ptx_thread_info* thread, int tid_in_warp) {
	g_cartographerts->onTxnReceivedAllCUReplies(core, wid, commit_warp_info, tx, thread, tid_in_warp);
}

void CartographerTimeSeries::do_removeInFlightCIDEntry(shader_core_ctx* core, unsigned wid, unsigned cid,
		unsigned tid_in_warp) {
	for (std::vector<MyCommittingTxnEntry*>::iterator itr = txn_cid_in_flight.begin();
			itr != txn_cid_in_flight.end(); itr++) {
		if ((*itr)->core == core &&
			(*itr)->wid == wid &&
			(*itr)->cid == cid) {
			MyCommittingTxnEntry* x = *itr;
			txn_cid_done.push_back(x);
			txn_cid_in_flight.erase(itr);
			break;
		}
	}

	for (std::vector<MyCommittingTxnEntry*>::iterator itr = txn_cid_in_flight.begin();
		itr != txn_cid_in_flight.end(); itr++) {
		if ((*itr)->core == core &&
			(*itr)->wid == wid &&
			(*itr)->cid == cid &&
			(*itr)->tid_in_warp == tid_in_warp) {
			printf("The same warp has two in-flight committing records, which could not be correct!\n");
			assert(0);
		}
	}
}

void CartographerTimeSeries::onTommyPacketSent(shader_core_ctx* core, unsigned wid, unsigned cid, unsigned tid_in_warp) {
	do_removeInFlightCIDEntry(core, wid, cid, tid_in_warp);
}

void CartographerTimeSeries::onTxnReceivedAllCUReplies(shader_core_ctx* core, unsigned wid,
		struct tx_log_walker::warp_commit_tx_t* commit_warp_info, struct tx_log_walker::commit_tx_t* tx,
		const ptx_thread_info* thread, int tid_in_warp) {
	if (!account_committing_warps) return;

	unsigned cid = tx->m_commit_id;
	if (dump_txn_rwsets) {
		unsigned long long cycle = getCurrentCycleCount();
		const char enter_exit = 0;

		gzwrite(f_rwsets, &cycle, sizeof(unsigned long long));
		gzwrite(f_rwsets, &enter_exit, sizeof(char));
		dim3 ctaid3 = thread->get_ctaid();
		dim3 tid3   = thread->get_tid();
		gzwrite(f_rwsets, &ctaid3, sizeof(dim3));
		gzwrite(f_rwsets, &tid3,   sizeof(dim3));


		char rw = 'R'; unsigned sz = read_sets.at(cid).size();
		gzwrite(f_rwsets, &rw, sizeof(char));
		gzwrite(f_rwsets, &sz, sizeof(unsigned));

		addr_set_t::iterator itr = read_sets.at(cid).begin();
		for (; itr != read_sets.at(cid).end(); itr++) {
			unsigned addr = *itr;
			gzwrite(f_rwsets, &addr, sizeof(unsigned));
		}

		rw = 'W'; sz = write_sets.at(cid).size();
		gzwrite(f_rwsets, &rw, sizeof(char));
		gzwrite(f_rwsets, &sz, sizeof(unsigned));
		itr = write_sets.at(cid).begin();
		for (; itr != write_sets.at(cid).end(); itr++) {
			unsigned addr = *itr;
			gzwrite(f_rwsets, &addr, sizeof(unsigned));
		}

		read_sets.erase(cid);
		write_sets.erase(cid);
	}

	do_removeInFlightCIDEntry(core, wid, cid, tid_in_warp);

	core->m_thread[wid * core->m_warp_size + tid_in_warp]->m_is_committing = false;
}

void CartographerTimeSeries_printAllReadSets() {
	g_cartographerts->printAllReadSets();
}
void CartographerTimeSeries::printAllReadSets() {
	printf("[CartographerTimeSeries] printAllReadSets\n");
	for (std::map<unsigned, addr_set_t>::iterator itr = read_sets.begin();
		itr != read_sets.end(); itr++) {
		printReadSet(itr->first);
	}
}

void CartographerTimeSeries_printReadSet(unsigned int cid) {
	g_cartographerts->printReadSet(cid);
}
void CartographerTimeSeries::printReadSet(unsigned int cid) {
	if (read_sets.find(cid) == read_sets.end()) {
		printf("[CartographerTimeSeries] Read set of cid=%u is empty.\n", cid);
	} else {
		addr_set_t* as = &(read_sets.at(cid));
		printf("[CartographerTimeSeries] Read set of cid=%u:", cid);
		for (addr_set_t::iterator itr = as->begin(); itr != as->end(); itr++) {
			printf(" %08x", *itr);
		}
		printf("\n");
	}
}

void CartographerTimeSeries_getCommittingTxnsInfo_2(std::vector<shader_core_ctx*>* cores, std::vector<unsigned>* wids,
		std::vector<struct tx_log_walker::warp_commit_tx_t>* commit_warps, std::vector<unsigned>* cids,
		std::vector<unsigned long long>* atimes,
		bool is_done_fill_only) {
	g_cartographerts->getCommittingTxnsInfo_2(cores, wids, commit_warps, cids, atimes, is_done_fill_only);
}
void CartographerTimeSeries::getCommittingTxnsInfo_2(std::vector<shader_core_ctx*>* cores, std::vector<unsigned>* wids,
		std::vector<struct tx_log_walker::warp_commit_tx_t>* commit_warps, std::vector<unsigned>* cids,
		std::vector<unsigned long long>* atimes,
		bool is_done_fill_only) {
	for (std::vector<MyCommittingTxnEntry*>::iterator itr = txn_cid_in_flight.begin();
				itr != txn_cid_in_flight.end(); itr ++) {
		if (is_done_fill_only) { if ((*itr)->has_done_fill == false) continue; }
		cores       ->push_back((*itr)->core);
		wids        ->push_back((*itr)->wid);
		commit_warps->push_back((*itr)->commit_warp_info);
		cids        ->push_back((*itr)->cid);
		atimes      ->push_back((*itr)->send_rs_started);
	}
}

void CartographerTimeSeries_getCommittingTxnsInfo(std::vector<shader_core_ctx*>* cores, std::vector<unsigned>* wids,
		std::vector<struct tx_log_walker::warp_commit_tx_t>* commit_warps, bool is_done_fill_only) {
	g_cartographerts->getCommittingTxnsInfo(cores, wids, commit_warps, is_done_fill_only);
}
void CartographerTimeSeries::getCommittingTxnsInfo(std::vector<shader_core_ctx*>* cores, std::vector<unsigned>* wids,
		std::vector<struct tx_log_walker::warp_commit_tx_t>* commit_warps, bool is_done_fill_only) {
	assert (account_committing_warps && "Must enable accounting of currently-committing warps");
	for (std::vector<MyCommittingTxnEntry*>::iterator itr = txn_cid_in_flight.begin();
			itr != txn_cid_in_flight.end(); itr ++) {
		if (is_done_fill_only) { if ((*itr)->has_done_fill == false) continue; }
		cores       ->push_back((*itr)->core);
		wids        ->push_back((*itr)->wid);
		commit_warps->push_back((*itr)->commit_warp_info);
	}
}

void CartographerTimeSeries_getCommitTxInfoFromCommitID(unsigned cid,
		tx_log_walker::warp_commit_tx_t** txt, shader_core_ctx** core, unsigned long long* atime) {
	return g_cartographerts->getCommitTxInfoFromCommitID(cid, txt, core, atime);
}
void CartographerTimeSeries::getCommitTxInfoFromCommitID(unsigned cid,
		tx_log_walker::warp_commit_tx_t** txt, shader_core_ctx** core, unsigned long long* atime) {
	for (std::vector<MyCommittingTxnEntry*>::iterator itr = txn_cid_in_flight.begin();
			itr != txn_cid_in_flight.end(); itr ++) {
		if ((*itr)->cid == cid) {
			*txt = &((*itr)->commit_warp_info);
			*core = (*itr)->core;
			*atime = (*itr)->send_rs_started;
			return;
		}
	}
	*txt = NULL;
	*core = NULL;
}

enum CartographerTimeSeries::MyThreadState CartographerTimeSeries::getMyThreadState (ptx_thread_info* ptx_thd) {
	return (enum MyThreadState) 0;
}

void CartographerTimeSeries_accountAllThreads() {
	g_cartographerts->accountAllThreads();
}
void CartographerTimeSeries::accountAllThreads() {
	if (!account_every_cycle) return;
//	std::map<enum MyThreadState, unsigned> hist;
	unsigned count[(int)(NUM_MYTHREADSTATE)];
	for (int i=0; i<(int)(NUM_MYTHREADSTATE); i++) count[i] = 0;

	for (unsigned i=0;i < g_the_gpu->m_shader_config->n_simt_clusters; i++) {
		simt_core_cluster* clus = g_the_gpu->m_cluster[i];

		for (std::list<unsigned>::iterator it = clus->m_core_sim_order.begin();
			it != clus->m_core_sim_order.end(); ++it) {
			shader_core_ctx* scc = clus->m_core[*it];
			for (unsigned wid = 0; wid < scc->m_warp_count; wid ++) {
				simt_stack* stack = scc->m_simt_stack[wid];
				if (stack->m_stack.size() > 0) {
					// Stolen from scheduler::cycle
					const active_mask_t &active_mask =
						scc->m_simt_stack[wid]->get_active_mask();
					for (unsigned t = 0; t < scc->m_warp_size; t++) {
						unsigned tid = t + wid * scc->m_warp_size;
						ptx_thread_info* pi = scc->m_thread[tid];
						{
							enum MyThreadState state;
							if (active_mask.test(t)) {
								state = THD_RUNNING;
								tm_manager_inf* pit = (pi->is_in_transaction());
								if (pit) {
									state = THD_TXN_EXEC;
									if (pi->m_is_committing == true) {
										state = THD_TXN_COMMITTING;
									}
								}
							} else {
								state = THD_INACTIVE;
							}

							if (pi) {
								dim3 ctaid3 = pi->get_ctaid(), tid3 = pi->get_tid();
								CTAID_TID_Ty ctaidtid(ctaid3, tid3);

								if (thd_cycle_breakdown_ref.find(ctaidtid) == thd_cycle_breakdown_ref.end()) {
									thd_cycle_breakdown_ref[ctaidtid] = std::map<std::string, unsigned>();
									thd_cycle_breakdown_ref[ctaidtid]["active"] = 0;
									thd_cycle_breakdown_ref[ctaidtid]["inactive"] = 0;
								}

								switch (state) {
								case THD_RUNNING:
								case THD_TXN_EXEC:
								case THD_TXN_COMMITTING:
									thd_cycle_breakdown_ref[ctaidtid]["active"] ++; break;
								case THD_INACTIVE:
									thd_cycle_breakdown_ref[ctaidtid]["inactive"] ++; break;
								default:
									assert(0);
								}
/*
								if (state == THD_RUNNING)
									thd_cycle_breakdown_ref[ctaidtid]["exec"] ++;
								if (state == THD_TXN_COMMITTING)
									thd_cycle_breakdown_ref[ctaidtid]["commit"] ++;*/

								if (prev_thd_state.find(ctaidtid) != prev_thd_state.end()) {
									enum MyThreadState prev_state = prev_thd_state[ctaidtid];
									if (prev_state != state) {
										struct MyThreadStateChangeTy mtsct = {};
										mtsct.state0 = prev_state;
										mtsct.state1 = state;
										mtsct.cycle = getCurrentCycleCount();
										if (thd_state_change_history.find(ctaidtid) == thd_state_change_history.end()) {
											thd_state_change_history[ctaidtid] = std::list<struct MyThreadStateChangeTy>();
										}
										thd_state_change_history[ctaidtid].push_back(mtsct);
									}
								}
								prev_thd_state[ctaidtid] = state;

							}

							count[(int)state] ++;
//							if (hist.find(state) == hist.end()) hist[state] = 0;
//							hist[state] += 1;
						}
					}
				} else {
				}
			}
		}
	}

	bool is_insert = true;
	std::map<enum MyThreadState, unsigned> hist;
	for (int i=0; i<(int)(NUM_MYTHREADSTATE); i++) {
		hist[(enum MyThreadState)i] = count[i];
	}
	if (thd_status_history_ref.size() > 0) {
		if (thd_status_history_ref.back().second == hist) is_insert = false;
	}
	if (is_insert) {
		unsigned cyc = getCurrentCycleCount();
		thd_status_history_ref.push_back(std::make_pair(cyc, hist));
	}
}

void CartographerTimeSeries::dumpThreadStateHist() {
	int err; sqlite3* db;
	const char* status_str[] = {
		"INACTIVE",
		"RUNNING",
		"COMMITTING"
	};
	if ((err = sqlite3_open("thd_status_hist.db", &db)) != SQLITE_OK) {
			printf("Error code: %d\n", err);
			assert (0 && "SQLite database open error!");
		} else {
			sqlite3_stmt* drop_stmt, *create_stmt, *insert_stmt;
			const char* drop_query = "DROP TABLE IF EXISTS thd_state_hist;";
			sqlite3_prepare_v2(db, drop_query, (int)strlen(drop_query), &drop_stmt, NULL);
			sqlite3_step(drop_stmt);
			sqlite3_finalize(drop_stmt);

			std::string create_query = "CREATE TABLE IF NOT EXISTS thd_state_hist (Cycle INTEGER";
			for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
				create_query += ", ";
				create_query += status_str[i];
				create_query += " INTEGER";
			}
			create_query += ");";
			sqlite3_prepare_v2(db, create_query.c_str(), (int)(create_query.size()), &create_stmt, NULL);
			if (!(sqlite3_step(create_stmt) == SQLITE_DONE)) {
				printf("[CartographerMem] Oh! cannot create table!\n");
				return;
			}
			sqlite3_finalize(create_stmt);

			std::string insert_query = "INSERT INTO thd_state_hist (Cycle";
			for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
				insert_query += ", ";
				insert_query += status_str[i];
			}
			insert_query += ") VALUES (?";
			for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
				insert_query += " ,?";
			}
			insert_query += ");";

			struct timeval tick, tock;
			gettimeofday(&tick, NULL);
			unsigned num_ety = 0;

			// Insert the entries!!!
			sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
			std::deque<std::pair<unsigned long long, std::map<enum MyThreadState, unsigned> > >::iterator itr =
				thd_status_history_ref.begin();
			for (; itr != thd_status_history_ref.end(); itr++) {
				err = sqlite3_prepare_v2 (db, insert_query.c_str(), -1, &insert_stmt, NULL);
				if (err != SQLITE_OK) {
					printf("Insert SQL Query: %s\n", insert_query.c_str());
					printf("[CartographerTimeSeries] Err = %d !!\n", err); exit(1);
				}

				unsigned long long cyc = itr->first;
				std::map<enum MyThreadState, unsigned>* hist = &(itr->second);
				err = sqlite3_bind_int(insert_stmt, 1, cyc);
				if (err != SQLITE_OK) { printf("[CartographerTimeSeries] (1) In binding int err=%d\n", err); exit(1); }
				for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
					unsigned valu = 0;
					enum MyThreadState key = (enum MyThreadState) i;
					if (hist->find(key) != hist->end()) {
						valu = hist->at(key);
					}
					err = sqlite3_bind_int(insert_stmt, 2+i, valu);
					if (err != SQLITE_OK) { printf("[CartographerTimeSeries] (%d) In binding int err=%d\n", 2+i, err); exit(1); }
				}
				err = sqlite3_step(insert_stmt);
				if (err != SQLITE_DONE) {
					printf("[CartographerTimeSeries] Can't insert!\n");
					exit(1);
				}
				num_ety ++;
				sqlite3_finalize(insert_stmt);
			}
			sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
			sqlite3_close(db);

			gettimeofday(&tock, NULL);
			unsigned long tdiff = (tock.tv_sec - tick.tv_sec) * 1000 + (tock.tv_usec - tick.tv_usec) / 1000;
			fprintf(stderr, "Inserted %u entries in %lu milliseconds\n", num_ety, tdiff);
			fprintf(stderr, "End database transaction\n");
		}
}

void CartographerTimeSeries_onPTXInitThread(core_t* core, dim3 ctaid, dim3 tid, unsigned wid) {
	g_cartographerts->onPTXInitThread(core, ctaid, tid, wid);
}
void CartographerTimeSeries::onPTXInitThread(core_t* core, dim3 ctaid, dim3 tid, unsigned wid) {
	if (!log_events) return;
	CTAID_TID_Ty the_id(ctaid, tid);
	MyThreadInitEvent* evt = new MyThreadInitEvent(getCurrentCycleCount(), wid);
	appendEvent(the_id, evt);
	unsigned sid = 0;
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(core);
	if (scc) sid = scc->m_sid;
	std::pair<unsigned, unsigned> sid_wid = std::make_pair(sid, wid);
	if (sid_wid_to_threads.find(sid_wid) == sid_wid_to_threads.end()) {
		sid_wid_to_threads[sid_wid] = std::set<CTAID_TID_Ty>();
	}
	sid_wid_to_threads.at(sid_wid).insert(the_id);
}

void CartographerTimeSeries_onThreadExit(ptx_thread_info* pti) {
	g_cartographerts->onThreadExit(pti);
}
void CartographerTimeSeries::onThreadExit(ptx_thread_info* pti) {
	if (!log_events) return;
	CTAID_TID_Ty the_id(pti->get_ctaid(), pti->get_tid());
	if (dynamic_cast<MyThreadExitEvent*>(getCurrEvent(the_id)) == NULL) {
		MyThreadEvent* evt = new MyThreadExitEvent(getCurrentCycleCount());
		appendEvent(the_id, evt);
	}
}

void CartographerTimeSeries_onStartScoreboardCommit(shader_core_ctx* core, unsigned warp_id, warp_inst_t* inst) {
	g_cartographerts->onStartScoreboardCommit(core, warp_id, inst);
}
void CartographerTimeSeries::onStartScoreboardCommit(shader_core_ctx* core, unsigned warp_id, warp_inst_t* inst) {
	if (!log_events) return;
	unsigned warp_size = core->m_warp_size;
	for (unsigned i=0; i<warp_size; i++) {
		if (inst->active(i)) {
			unsigned tidx = warp_id * warp_size + i;
			ptx_thread_info* pti = core->m_thread[tidx];
			dim3 ctaid = pti->get_ctaid(), tid = pti->get_tid();
			CTAID_TID_Ty the_id (ctaid, tid);
			appendEvent(the_id, new MyThreadStartScoreboardCommitEvent(getCurrentCycleCount()));
		}
	}
}


void CartographerTimeSeries_onDoneScoreboardCommit(shader_core_ctx* core, unsigned warp_id, warp_inst_t* inst) {
	g_cartographerts->onDoneScoreboardCommit(core, warp_id, inst);
}
void CartographerTimeSeries::onDoneScoreboardCommit(shader_core_ctx* core, unsigned warp_id, warp_inst_t* inst) {
	if (!log_events) return;
	unsigned warp_size = core->m_warp_size;
	for (unsigned i=0; i<warp_size; i++) {
		if (inst->active(i)) {
			bool is_passed = false;
			unsigned tidx = warp_id * warp_size + i;
			ptx_thread_info* pti = core->m_thread[tidx];

			dim3 ctaid = pti->get_ctaid(), tid = pti->get_tid();
			CTAID_TID_Ty the_id (ctaid, tid);
			appendEvent(the_id, new MyThreadDoneScoreboardCommitEvent(getCurrentCycleCount(), is_passed));
		}
	}
}

// Failing Pre-Commit Validation would also put the active mask to zero, as is shown in
// void tx_log_walker::intra_warp_conflict_detection(warp_inst_t &inst, iwcd_uarch_info &uarch_activity).
void CartographerTimeSeries_onThreadFailedPreCommitValidation(shader_core_ctx* core, ptx_thread_info* pti, const char* _reason) {
	g_cartographerts->onThreadFailedPreCommitValidation(core, pti, _reason);
}
void CartographerTimeSeries::onThreadFailedPreCommitValidation(shader_core_ctx* core, ptx_thread_info* pti, const char* _reason) {
	if (!log_events) return;
	dim3 ctaid = pti->get_ctaid(), tid = pti->get_tid();
	CTAID_TID_Ty the_id (ctaid, tid);
	if (dynamic_cast<MyThreadFailPreCommitValidation*>(getCurrEvent(the_id))) return; // May be duplicate PCV Fail entries in one cycle; de-dup
	appendEvent(the_id, new MyThreadFailPreCommitValidation(getCurrentCycleCount(), _reason));
	appendEvent(the_id, new MyThreadActiveMaskChangedEvent(getCurrentCycleCount(), "Failed PreCommitValidation", 0));
}

void CartographerTimeSeries_onThreadDoneCoreSideCommit (shader_core_ctx* core, ptx_thread_info* pti) {
	g_cartographerts->onThreadDoneCoreSideCommit(core, pti);
}
void CartographerTimeSeries::onThreadDoneCoreSideCommit(shader_core_ctx* core, ptx_thread_info* pti) {
	if (!log_events) return;
	dim3 ctaid = pti->get_ctaid(), tid = pti->get_tid();
	CTAID_TID_Ty the_id (ctaid, tid);
	appendEvent(the_id, new MyThreadDoneCoreSideReadonlyCommitEvent(getCurrentCycleCount()));
}

void CartographerTimeSeries_onRemoveAllWarpToTIDMapOnCore(core_t* core) {
	g_cartographerts->onRemoveAllWarpToTIDMapOnCore(core);
}
// May exit more than once
void CartographerTimeSeries::onRemoveAllWarpToTIDMapOnCore(core_t* core) {
	if (!log_events) return;
	unsigned sid = 0;
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*> (core);
	if (scc) sid = scc->m_sid;
	while (true) {
		bool is_found = false;
		for (std::map<std::pair<unsigned, unsigned>, std::set<CTAID_TID_Ty> >::iterator itr =
			sid_wid_to_threads.begin(); itr != sid_wid_to_threads.end(); itr++) {
			std::pair<unsigned, unsigned> sid_wid = itr->first;
			if (sid_wid.first == sid) { itr = sid_wid_to_threads.erase(itr); is_found = true; break; }
		}
		if (is_found == false) break;
	}
	printf("[CartographerTimeSeries] Remove all Warp to TID Map (S%u)\n", sid);
}

void CartographerTimeSeries_onWarpScoreboardHazard (shader_core_ctx* core, unsigned wid, const char* reason) {
	g_cartographerts->onWarpScoreboardHazard(core, wid, reason);
}
void CartographerTimeSeries::onWarpScoreboardHazard(shader_core_ctx* core, unsigned wid, const char* reason) {
	if (!log_events) return;
	unsigned sid = core->m_sid;
	std::pair<unsigned, unsigned> sid_wid = std::make_pair(sid, wid);
	assert (sid_wid_to_threads.find(sid_wid) != sid_wid_to_threads.end());
	std::set<CTAID_TID_Ty>* ctt = &(sid_wid_to_threads.at(sid_wid));
	const simt_mask_t* mask = &(core->m_simt_stack[wid]->get_active_mask());
	unsigned t = 0;
	for (std::set<CTAID_TID_Ty>::iterator itr = ctt->begin(); itr != ctt->end(); itr++, t++) {
		if (mask->test(t) == 0) continue;
		CTAID_TID_Ty ct = *itr;
		MyThreadEvent* last = getCurrEvent(ct);
		MyThreadScoreboardHazardEvent* sbhe = dynamic_cast<MyThreadScoreboardHazardEvent*>(last);
		if (sbhe) {
			if (sbhe->reason == reason) sbhe->until = getCurrentCycleCount();
			else appendEvent(ct, (new MyThreadScoreboardHazardEvent(getCurrentCycleCount(), reason)));
		} else {
			appendEvent(ct, (new MyThreadScoreboardHazardEvent(getCurrentCycleCount(), reason)));
		}
	}
}

void CartographerTimeSeries_onThreadTLWStateChanged(shader_core_ctx* core, ptx_thread_info* pti, enum tx_log_walker::commit_tx_state_t _state) {
	g_cartographerts->onThreadTLWStateChanged(core, pti, _state);
}
void CartographerTimeSeries::onThreadTLWStateChanged(shader_core_ctx* core, ptx_thread_info* pti, enum tx_log_walker::commit_tx_state_t _state) {
	if (!log_events) return;
	dim3 ctaid = pti->get_ctaid(), tid = pti->get_tid();
	CTAID_TID_Ty the_id (ctaid, tid);
	appendEvent(the_id, new MyThreadTLWStateChangedEvent(getCurrentCycleCount(), _state));
}

void CartographerTimeSeries_onThreadSendTxPassFail(shader_core_ctx* core, ptx_thread_info* pti, enum tx_log_walker::commit_tx_state_t _state, bool is_pass) {
	g_cartographerts->onThreadSendTxPassFail(core, pti, _state, is_pass);
}
void CartographerTimeSeries::onThreadSendTxPassFail(shader_core_ctx* core, ptx_thread_info* pti, enum tx_log_walker::commit_tx_state_t _state, bool is_pass) {
	if (!log_events) return;
	assert (_state == tx_log_walker::SEND_ACK_CLEANUP);
	dim3 ctaid = pti->get_ctaid(), tid = pti->get_tid();
	CTAID_TID_Ty the_id (ctaid, tid);
	MyThreadTLWStateChangedEvent* evt = new MyThreadTLWStateChangedEvent(getCurrentCycleCount(), _state);
	DBG(printf("[CartographerTimeSeries] Thread (%u,%u,%u)-(%u,%u,%u) sends is_pass=%d @ cycle %llu\n",
		ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z, is_pass, getCurrentCycleCount()));
	evt->is_passed = is_pass;
	appendEvent(the_id, evt);
}

void CartographerTimeSeries_onWarpActiveMaskChanged(core_t* core, unsigned warp_id, const simt_mask_t* mask, const char* _reason) {
	g_cartographerts->onWarpActiveMaskChanged(core, warp_id, mask, _reason);
}
void CartographerTimeSeries::onWarpActiveMaskChanged(core_t* core, unsigned warp_id, const simt_mask_t* mask, const char* _reason) {
	if (!log_events) return;
	shader_core_ctx* scc = dynamic_cast<shader_core_ctx*>(core);
//	unsigned sid = (unsigned)(-1);
//	if (scc) sid = scc->m_sid;
//	DBG(printf("[CartographerTimeSeries] S%u,W%u active mask changed, reason is %s @ cycle %llu\n", sid, warp_id, _reason, getCurrentCycleCount()));
	for (unsigned i=0; i<core->m_warp_size; i++) {
		unsigned t_tid = i + warp_id * core->m_warp_size;
		ptx_thread_info* pti = core->m_thread[t_tid];
		if (pti) {
			dim3 ctaid = pti->get_ctaid(), tid = pti->get_tid();
			CTAID_TID_Ty the_id (ctaid, tid);

			MyThreadActiveMaskChangedEvent* prev = dynamic_cast<MyThreadActiveMaskChangedEvent*>(getCurrEvent(the_id));
			if (prev) { if (prev->changed_to == mask->test(i)) continue; }
			MyThreadActiveMaskChangedEvent* evt = new MyThreadActiveMaskChangedEvent(getCurrentCycleCount(),
					_reason, mask->test(i));
			appendEvent(the_id, evt);
		}
	}
}

#define MF_TUP_BEGIN(x) static std::map<enum x, std::string> MEM_FETCH_STR = {
#define MF_TUP(x) { x, #x }
#define MF_TUP_END(x) };
#include "gpgpu-sim/mem_fetch_status.tup"
#undef MF_TUP_END
#undef MF_TUP
#undef MF_TUP_BEGIN

void CartographerMem::do_onNewMemFetch(shader_core_ctx* shader, warp_inst_t* inst, mem_fetch* mf, bool is_l2) {
	MyMemFetchInfo mmfi = {};
	if (inst->empty()) {
		mmfi.warp_id = -999; // Instruction Cache. The warp_inst_t does not have warp_id for now.
	} else mmfi.warp_id = inst->warp_id();
	if (shader == NULL) {
		mmfi.shader_id = -999; // L2 Cache access
	} else mmfi.shader_id = shader->m_sid;
	mmfi.sent_to_icnt = getCurrentCycleCount();
	mmfi.mf = mf;
	mmfi.serial_number = mf->m_serial_number;

	{
		// Stolen from shader.cc,
		//  void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf) {
		unsigned int packet_size = mf->size();
			if (!mf->get_is_write() && !mf->isatomic()
					&& !(mf->get_access_type() == TX_MSG)) {
				packet_size = mf->get_ctrl_size();
			}
		if (!is_l2) { // In order to compare against GPGPUSim's icnt-side stats we have to ignore L2
			traffic_breakdown_sent->record_traffic(mf, packet_size);
			this->mem_fetches.push_back(mmfi);
			this->sent_count ++;
			if (is_dump_all) {
				printf("[CartographerMem] CORE->ICNT: ");
				mf->print(stdout, true);
			}
		}
	}
}

void CartographerMem_onMemFetchInCacheMissQueue(shader_core_ctx* shader, cache_t* cache, warp_inst_t* inst, mem_fetch* mf) {
	g_cartographermem->onMemFetchInCacheMissQueue(shader, cache, inst, mf);
}

void CartographerMem::onMemFetchInCacheMissQueue(shader_core_ctx* shader, cache_t* cache, warp_inst_t* inst, mem_fetch* mf) {
	bool is_l2 = false;
		if (dynamic_cast<l2_cache*> (cache) != NULL) is_l2 = true;
	do_onNewMemFetch(shader, inst, mf, is_l2);
}

void CartographerMem_onMemFetchInTLWMessageQueue (shader_core_ctx* shader, tx_log_walker* tlw,   warp_inst_t* inst, mem_fetch* mf) {
	g_cartographermem->onMemFetchInTLWMessageQueue(shader, tlw, inst, mf);
}
void CartographerMem::onMemFetchInTLWMessageQueue (shader_core_ctx* shader, tx_log_walker* tlw,   warp_inst_t* inst, mem_fetch* mf) {
	do_onNewMemFetch(shader, inst, mf, false);
}

void CartographerMem_onMemFetchInLDSTOutbound (shader_core_ctx* shader, ldst_unit* ldst, warp_inst_t* inst, mem_fetch* mf) {
	g_cartographermem->onMemFetchInLDSTOutbound(shader, ldst, inst, mf);
}
void CartographerMem::onMemFetchInLDSTOutbound (shader_core_ctx* shader, ldst_unit* ldst, warp_inst_t* inst, mem_fetch* mf) {
	do_onNewMemFetch(shader, inst, mf, false);
}

void CartographerMem_onMemFetchReceivedFromIcnt (mem_fetch* mf) {
	g_cartographermem->onMemFetchReceivedFromIcnt(mf);
}
void CartographerMem::onMemFetchReceivedFromIcnt(mem_fetch* mf) {
	std::deque<MyMemFetchInfo>::iterator itr = mem_fetches.begin();
	while (itr != mem_fetches.end()) {
		const MyMemFetchInfo* mmfi = &(*itr);
		if (mmfi->mf == mf || mmfi->serial_number == mf->m_serial_number) {
			itr = mem_fetches.erase(itr);
			if (is_log_mem_fetch_state_transition) {
				if (mf_prev_state.find(mf) != mf_prev_state.end())
					mf_prev_state.erase(mf);
			}
		} else itr++;
	}
	this->rcvd_count ++;

	// Stolen from shader.cc
	// void simt_core_cluster::icnt_cycle() {
	unsigned int packet_size =
					(mf->get_is_write()) ? mf->get_ctrl_size() : mf->size();
	traffic_breakdown_rcvd->record_traffic(mf, packet_size);
	if (is_dump_all) {
		printf("[CartographerMem] ICNT->CORE: ");
		mf->print(stdout, true);
	}
}

void CartographerMem_accountAllMemFetches() {
	g_cartographermem->accountAllMemFetches();
}
void CartographerMem::accountAllMemFetches() {
	if (!account_every_cycle) return;

	if ((getCurrentCycleCount() % 1000) == 0) {
		printf("[CartographerMem] mem_fetches.size() = %lu\n",
			mem_fetches.size());
	}

	std::map<enum mem_fetch_status, unsigned> h;
	std::deque<MyMemFetchInfo>::iterator itr = mem_fetches.begin();
	for (; itr != mem_fetches.end(); itr++) {
		const MyMemFetchInfo* mmfi = &(*itr);
		mem_fetch* mf = mmfi->mf;
		enum mem_fetch_status mat = mf->get_status();
		// Local histogram.
		if (h.find(mat) == h.end()) h[mat] = 0;
		h[mat] += 1;
		// Global cumulative histogram.
		if (mf_state_histo_cumsum.find(mat) == mf_state_histo_cumsum.end())
			mf_state_histo_cumsum[mat] = 0;
		mf_state_histo_cumsum[mat] += 1;

		if (is_log_mem_fetch_state_transition) {
			if (mf_prev_state.find(mf) != mf_prev_state.end()) {
				enum mem_fetch_status prev_st = mf_prev_state.at(mf);
				if (prev_st != mat) {
					std::pair<enum mem_fetch_status, enum mem_fetch_status> key = std::make_pair(prev_st, mat);
					if (mf_state_transition_count.find(key) == mf_state_transition_count.end()) {
						mf_state_transition_count[key] = 0;
					}
					mf_state_transition_count[key] += 1;
				}
			}
			mf_prev_state[mf] = mat;
		}
	}
	bool is_insert = true;
	if (mf_status_hist.size() > 0) {
		if (mf_status_hist.back().second == h) is_insert = false;
	}
	if (is_insert) {
		unsigned cyc = getCurrentCycleCount();
		mf_status_hist.push_back(std::make_pair(cyc, h));
	}
}

void CartographerMem::summary() {
	printf("[CartographerMem] Send/Received event tracked: %u vs %u\n", sent_count, rcvd_count);
	if (mem_fetches.size() == 0) {	printf("[CartographerMem] OK!\n");	}
	else {
		printf("[CartographerMem] There are still %lu live entries.\n", mem_fetches.size()); }
	traffic_breakdown_sent->print(stdout);
	traffic_breakdown_rcvd->print(stdout);
	printf("[CartographerMem] Mem Fetch Access Type State contains %lu entries.\n",
		mf_status_hist.size());

	if (account_every_cycle) {
		printf("Sum of num cycles spent in icnt states:\n");
		for (std::map<enum mem_fetch_status, unsigned long>::iterator
			itr = mf_state_histo_cumsum.begin(); itr != mf_state_histo_cumsum.end();
			itr ++) {
			if (MEM_FETCH_STR.find(itr->first) != MEM_FETCH_STR.end())
				printf("%s, %.2f\n", MEM_FETCH_STR.at(itr->first).c_str(), itr->second*1.0/rcvd_count);
			else
				printf("(Unknown State), %f\n", itr->second*1.0/rcvd_count);
		}

		if (is_log_mem_fetch_state_transition) {
			printf("Writing Mem Fetch State Transition Table to %s\n", fn_mem_fetch_state_transition.c_str());
			fprintf(f_mem_fetch_state_transition, "digraph G {\n");
			// Prepare keys.
			std::set<enum mem_fetch_status> all_states;
			for (std::map<std::pair<enum mem_fetch_status, enum mem_fetch_status>, unsigned>::iterator itr =
				mf_state_transition_count.begin(); itr != mf_state_transition_count.end(); itr++) {
				all_states.insert(itr->first.first);
				all_states.insert(itr->first.second);
			}
			for (std::set<enum mem_fetch_status>::iterator itr = all_states.begin(); itr != all_states.end();
					itr++) {
				if (MEM_FETCH_STR.find(*itr) == MEM_FETCH_STR.end()) continue; // Ugly!
				fprintf(f_mem_fetch_state_transition, "  Node%lu [ label=\"%s", (unsigned long)(*itr),
					MEM_FETCH_STR.find(*itr) == MEM_FETCH_STR.end() ?
						"(Unknown State)" : MEM_FETCH_STR.at(*itr).c_str());
				if (mf_state_histo_cumsum.find(*itr) != mf_state_histo_cumsum.end()) {
					fprintf(f_mem_fetch_state_transition, "\\navg %.2f cyc\" ];\n",
						mf_state_histo_cumsum.at(*itr)*1.0/rcvd_count);
				} else {
					fprintf(f_mem_fetch_state_transition, "\\navg ? cyc\" ];\n");
				}
			}
			for (std::map<std::pair<enum mem_fetch_status, enum mem_fetch_status>, unsigned>::iterator itr =
				mf_state_transition_count.begin(); itr != mf_state_transition_count.end(); itr++) {
				if (MEM_FETCH_STR.find(itr->first.first) == MEM_FETCH_STR.end()) continue; // Ugly!
				if (MEM_FETCH_STR.find(itr->first.second) == MEM_FETCH_STR.end()) continue; // Ugly!

				fprintf(f_mem_fetch_state_transition, "  Node%lu -> Node%lu [ label=\"%u\" ]\n",
						(unsigned long)(itr->first.first), (unsigned long)(itr->first.second), itr->second);
			}
			fprintf(f_mem_fetch_state_transition, "};\n");
		}

		dumpMemFetchHist();
	}
}

void CartographerMem::dumpMemFetchHist() {
	int err; sqlite3* db;
	if ((err = sqlite3_open(mem_hist_db_filename, &db)) != SQLITE_OK) {
		printf("Error code: %d\n", err);
		assert (0 && "SQLite database open error!");
	} else {
		sqlite3_stmt* drop_stmt, *create_stmt, *insert_stmt;
		const char* drop_query = "DROP TABLE IF EXISTS mem_fetch_hist;";
		sqlite3_prepare_v2(db, drop_query, (int)strlen(drop_query), &drop_stmt, NULL);
		sqlite3_step(drop_stmt);
		sqlite3_finalize(drop_stmt);

		std::string create_query = "CREATE TABLE IF NOT EXISTS mem_fetch_hist (Cycle INTEGER";
		for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
			create_query += ", ";
			create_query += status_str[i];
			create_query += " INTEGER";
		}
		create_query += ");";
		sqlite3_prepare_v2(db, create_query.c_str(), (int)(create_query.size()), &create_stmt, NULL);
		if (!(sqlite3_step(create_stmt) == SQLITE_DONE)) {
			printf("[CartographerMem] Oh! cannot create table!\n");
			return;
		}
		sqlite3_finalize(create_stmt);

		std::string insert_query = "INSERT INTO mem_fetch_hist (Cycle";
		for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
			insert_query += ", ";
			insert_query += status_str[i];
		}
		insert_query += ") VALUES (?";
		for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
			insert_query += " ,?";
		}
		insert_query += ");";

		struct timeval tick, tock;
		gettimeofday(&tick, NULL);
		unsigned num_ety = 0;

		// Insert the entries!!!
		sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
		std::list<std::pair<unsigned long long, std::map<enum mem_fetch_status, unsigned> > >::iterator itr =
			mf_status_hist.begin();
		for (; itr != mf_status_hist.end(); itr++) {
			err = sqlite3_prepare_v2 (db, insert_query.c_str(), -1, &insert_stmt, NULL);
			if (err != SQLITE_OK) {
				printf("Insert SQL Query: %s\n", insert_query.c_str());
				printf("[CartographerMem] Err = %d !!\n", err); exit(1);
			}

			unsigned long long cyc = itr->first;
			std::map<enum mem_fetch_status, unsigned>* hist = &(itr->second);
			err = sqlite3_bind_int(insert_stmt, 1, cyc);
			if (err != SQLITE_OK) { printf("[CartographerMem] (1) In binding int err=%d\n", err); exit(1); }
			for (unsigned i=0; i<sizeof(status_str)/sizeof(char*); i++) {
				unsigned valu = 0;
				enum mem_fetch_status key = (enum mem_fetch_status) i;
				if (hist->find(key) != hist->end()) {
					valu = hist->at(key);
				}
				err = sqlite3_bind_int(insert_stmt, 2+i, valu);
				if (err != SQLITE_OK) { printf("[CartographerMem] (%d) In binding int err=%d\n", 2+i, err); exit(1); }
			}
			err = sqlite3_step(insert_stmt);
			if (err != SQLITE_DONE) {
				printf("[CartographerMem] Can't insert!\n");
				exit(1);
			}
			num_ety ++;
			sqlite3_finalize(insert_stmt);
		}
		sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
		sqlite3_close(db);

		gettimeofday(&tock, NULL);
		unsigned long tdiff = (tock.tv_sec - tick.tv_sec) * 1000 + (tock.tv_usec - tick.tv_usec) / 1000;
		fprintf(stderr, "Inserted %u entries in %lu milliseconds\n", num_ety, tdiff);
		fprintf(stderr, "End database transaction\n");
	}
}

const char* getCommitTxStateString(enum tx_log_walker::commit_tx_state_t x) {
	switch (x) {
	case tx_log_walker::IDLE: return "IDLE";
	case tx_log_walker::INTRA_WARP_CD: return "INTRA_WARP_CD";
	case tx_log_walker::ACQ_CID: return "ACQ_CID";
	case tx_log_walker::ACQ_CU_ENTRIES: return "ACQ_CU_ENTRIES";
	case tx_log_walker:: WAIT_CU_ALLOC_REPLY: return "WAIT_CU_ALLOC_REPLY";
	case tx_log_walker::SEND_RS: return "SEND_RS";
	case tx_log_walker::SEND_WS: return "SEND_WS";
	case tx_log_walker::WAIT_CU_REPLY: return "WAIT_CU_REPLY";
	case tx_log_walker::RESEND_RS: return "RESEND_RS";
	case tx_log_walker::RESEND_WS: return "RESEND_WS";
	case tx_log_walker::SEND_ACK_CLEANUP: return "SEND_ACK_CLEANUP";
	default: return "I don't know!";
	}
}

const char* getCommitEntryStateString(enum commit_state s) {
	switch (s) {
	case UNUSED: return "UNUSED";
	case FILL: return "FILL";
	case HAZARD_DETECT: return "HAZARD_DETECT";
	case VALIDATION_WAIT: return "VALIDATION_WAIT";
	case REVALIDATION_WAIT: return "REVALIDATION_WAIT";
	case PASS: return "PASS";
	case FAIL: return "FAIL";
	case PASS_ACK_WAIT: return "PASS_ACK_WAIT";
	case COMMIT_READY: return "COMMIT_READY";
	case COMMIT_SENT: return "COMMIT_SENT";
	case RETIRED: return "RETIRED";
	default: return "I don't know!";
	}
}

void CartographerTimeSeries_dumpAll() {
	g_cartographerts->dumpAll();
}

extern std::unordered_set<unsigned> g_tommy_aborted_cids;
void CartographerTimeSeries::dumpAll() {
	for (unsigned i=0;i < g_the_gpu->m_shader_config->n_simt_clusters; i++) {
		simt_core_cluster* clus = g_the_gpu->m_cluster[i];

		for (std::list<unsigned>::iterator it = clus->m_core_sim_order.begin();
			it != clus->m_core_sim_order.end(); ++it) {
			shader_core_ctx* scc = clus->m_core[*it];
								tx_log_walker* tlw = scc->m_ldst_unit->m_TLW;
			for (unsigned wid = 0; wid < scc->m_warp_count; wid ++) {
				simt_stack* stack = scc->m_simt_stack[wid];
				if (stack->m_stack.size() > 0) {
					// Stolen from scheduler::cycle
					for (unsigned t = 0; t < scc->m_warp_size; t++) {
						unsigned tid = t + wid * scc->m_warp_size;
						ptx_thread_info* pi = scc->m_thread[tid];
						{
							if (pi) {
								dim3 ctaid3 = pi->get_ctaid(), tid3 = pi->get_tid();
								CTAID_TID_Ty ctaidtid(ctaid3, tid3);

								tm_manager_inf* tmi = pi->get_tm_manager();
								tm_manager* tm = dynamic_cast<tm_manager*>(tmi);
								if (tm) {
									/*
									printf("Thread (%d,%d,%d)-(%d,%d,%d) has tm_manager.\n",
										ctaid3.x, ctaid3.y, ctaid3.z, tid3.x, tid3.y, tid3.z);
										*/
								}
							}
						}
					}
				}

			}

			for (unsigned i=0; i<tlw->m_committing_warp.size(); i++) {
				tx_log_walker::warp_commit_tx_t* wct = &(tlw->m_committing_warp[i]);
				if (wct->active()) {
					printf("TLW's m_committing_warp[%d]:\n", i);
					for (unsigned j=0; j<wct->m_thread_state.size(); j++) {
						tx_log_walker::commit_tx_t& tp = (*wct).m_thread_state[j];
						printf("  Thread[%d] cid=%d state = %s(%d)\n", j, tp.m_commit_id,
							getCommitTxStateString(tp.m_state), tp.m_state);
					}
				}
			}

			printf("LDST Unit of core sid = %d response FIFO size = %lu\n",
					scc->m_sid,
					scc->m_ldst_unit->m_response_fifo.size());
		}
	}

	for (unsigned i=0;i<g_the_gpu->m_memory_config->m_n_mem_sub_partition;i++) {
		memory_sub_partition* msp = g_the_gpu->m_memory_sub_partition[i];
		commit_unit* cu = msp->m_commit_unit;
		printf("Commit Entries in Commit Unit of Memory Subpartition %u:\n", i);
		for (unsigned j=0; j<cu->m_commit_entry_table.size(); j++) {
			commit_entry* ce = &(cu->m_commit_entry_table[j]);
			printf("  cid=%d, state=%s(%d), RS: ", ce->get_commit_id(),
				getCommitEntryStateString(ce->m_state), ce->m_state);
			const cu_access_set::linear_buffer_t* lb = &(ce->m_read_set.get_linear_buffer());
			for (cu_access_set::linear_buffer_t::const_iterator itr = lb->begin(); itr != lb->end(); itr++) {
				printf("%08llx", *itr);
			}
			printf("\n");
		}
		printf("Response Queue Length of Commit Unit of Memory Subpartition %u: len = %lu\n",
			i, cu->m_response_queue.size());
	}

	printf("Aborted CIDS:");
	std::set<unsigned> tmp; tmp.clear();
	for (std::unordered_set<unsigned>::iterator itr = g_tommy_aborted_cids.begin();
		itr != g_tommy_aborted_cids.end(); itr++) {
		tmp.insert(*itr);
	}
	for (std::set<unsigned>::iterator itr = tmp.begin(); itr != tmp.end(); itr++) {
		printf(" %u", *itr);
	}
	printf("\n");
}

// 2015-09-02: Last round of data-gathering prior to submitting to HPCA16
void Cartographer_onTMManagerStart(unsigned uid) { g_cartographer->onTMManagerStart(uid); }
void Cartographer::onTMManagerStart(unsigned uid) {
	if (!g_tommy_log_0902) return;
	do_insertTxUIDTimeStamp(uid, 'S');
}
void Cartographer_onTMManagerAbort(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi)
{ g_cartographer->onTMManagerAbort(uid, tmm, twi); }
void Cartographer::onTMManagerAbort(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi) {
	if (!g_tommy_log_0902) return;
	// Log RWSet Size1
	do_appendRWSetSize(uid, tmm, twi);
	do_insertTxUIDTimeStamp(uid, 'A');
}
void Cartographer_onTMManagerCommit(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi)
{ g_cartographer->onTMManagerCommit(uid, tmm, twi); }
void Cartographer::onTMManagerCommit(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi) {
	if (!g_tommy_log_0902) return;
	// Log at C also
	do_appendRWSetSize(uid, tmm, twi);
	do_insertTxUIDTimeStamp(uid, 'C');
}
void Cartographer_onTMManagerSendRS(unsigned uid) { g_cartographer->onTMManagerSendRS(uid); }
void Cartographer::onTMManagerSendRS(unsigned uid) {
	if (!g_tommy_log_0902) return;
	do_insertTxUIDTimeStamp(uid, 's');
}
void Cartographer_onTMManagerWaitCUReply(unsigned uid) { g_cartographer->onTMManagerWaitCUReply(uid); }
void Cartographer::onTMManagerWaitCUReply(unsigned uid) {
	if (!g_tommy_log_0902) return;
	do_insertTxUIDTimeStamp(uid, 'w');
}

void Cartographer::do_insertTxUIDTimeStamp(unsigned uid, char tag) {
	unsigned ts = getCurrentCycleCount();
	std::pair<char, unsigned> x = std::make_pair(tag, ts);
	if (uid_to_timestamps.find(uid) == uid_to_timestamps.end()) {
		uid_to_timestamps[uid] = std::vector<std::pair<char, unsigned> >();
	}
	uid_to_timestamps[uid].push_back(x);

	if (tag == 'S' || tag == 'A') {
		if (tag == 'A') {
			TxnEpochStates* x1 = &(uid_to_rwlogstats.at(uid).back());
			x1->cycle_end = getCurrentCycleCount();
			x1->outcome = 'A';
		}
		if (uid_to_rwlogstats.find(uid) == uid_to_rwlogstats.end())
			uid_to_rwlogstats[uid] = std::vector<TxnEpochStates>();
		TxnEpochStates x;
		x.cycle_start = getCurrentCycleCount();
		uid_to_rwlogstats[uid].push_back(x);
	} else if (tag == 's') {
		TxnEpochStates* x = &(uid_to_rwlogstats.at(uid).back());
		x->cycle_send_rw = getCurrentCycleCount();
	} else if (tag == 'w') {
		TxnEpochStates* x = &(uid_to_rwlogstats.at(uid).back());
		x->cycle_wait_cu_reply = getCurrentCycleCount();
	} else if (tag == 'C') {
		TxnEpochStates* x = &(uid_to_rwlogstats.at(uid).back());
		x->cycle_end = getCurrentCycleCount();
		x->outcome = 'C';
	}
}

void Cartographer_onTMManagerSendRWSetEntry(unsigned uid, char rw, unsigned addr_tag) {
	g_cartographer->onTMManagerSendRWSetEntry(uid, rw, addr_tag);
}
void Cartographer::onTMManagerSendRWSetEntry(unsigned uid, char rw, unsigned addr_tag) {
	do_appendTxUIDRWLogEntry(uid, rw, addr_tag);
}
void Cartographer::do_appendTxUIDRWLogEntry(unsigned uid, char rw, unsigned addr_tag) {
	if (g_tommy_log_0902 == false) return;
	assert (uid_to_rwlogstats.find(uid) != uid_to_rwlogstats.end());
	TxnEpochStates* x = &(uid_to_rwlogstats.at(uid).back());
	if (rw == 'R') {
		x->read_set_size ++;
		if (addr_tag != 0) x->read_set_size_nz ++;
	} else if (rw == 'W') {
		x->write_set_size ++;
		if (addr_tag != 0) x->write_set_size_nz ++;
	}
}

void Cartographer::do_appendRWSetSize(unsigned uid, tm_manager* tmm, struct tm_warp_info* twi) {
	assert (uid_to_rwlogstats.find(uid) != uid_to_rwlogstats.end());
	TxnEpochStates* x = &(uid_to_rwlogstats.at(uid).back());
	x->n_read = tmm->m_n_read;
	x->n_read_all = tmm->m_n_read_all;
	x->n_write = tmm->m_n_write;
	x->read_set_size_1 = tmm->m_read_word_set.size();
	x->write_set_size_1= tmm->m_write_word_set.size();
	if (twi) {
		x->read_set_size_2 = twi->m_read_log_size;
		x->write_set_size_2= twi->m_write_log_size;
	}
	x->read_word_set = tmm->m_read_word_set;
	x->write_word_set = tmm->m_write_word_set;
	ptx_thread_info* pti = tmm->m_thread;

	x->hw_sid = pti->get_hw_sid();
	x->hw_wid = pti->get_hw_wid();
	x->hw_tid = pti->get_hw_tid();

	x->ctaid = pti->get_ctaid();
	x->tid   = pti->get_tid();

	x->uid   = uid; // Txn's UID
	x->epoch = uid_to_rwlogstats.at(uid).size() - 1;
}

struct MyTxnEpochStats {
	unsigned long long start_cycle, sendlog_cycle, end_cycle;
	unsigned ws_size, ws_size_nz, rs_size, rs_size_nz;
	char outcome[10]; // Aborted or Committed
};

void Cartographer_DumpTMHistory() { g_cartographer->DumpTMHistory(); }
void Cartographer::DumpTMHistory() {
	if (g_tommy_log_0902) {
		printf("============ Logging information required on Sep 02 ============\n");
		int num_uid = 0, num_events = 0, num_epoches = 0;
		FILE* x = fopen("tx_all_timeline.txt", "w");
		FILE* f_rwsz = fopen("tx_rwset_size.txt", "w");
		assert(x);
		assert(f_rwsz);
		fprintf(x,      "UID\tTag\tCycle\n");
		// The first 4 are the R/W sets seen by the TLW
		fprintf(f_rwsz, "UID\tEpoch\tOutcome\tRSetSize\tRSetSizeNZ\tWSetSize\tWSetSizeNZ\tRSetSize1\tWSetSize1\n");
		unsigned long long sum_exec = 0,
							  sum_log_walk = 0,
							  sum_wait_cu = 0, // Time spent in EXECuting a Txn
							  sum_rset_size = 0, sum_rset_size_nz = 0,
							  sum_wset_size = 0, sum_wset_size_nz = 0,
							  sum_rset_size_1 = 0, sum_wset_size_1 = 0;
							  ;

		std::set<unsigned> all_uids;
		for (std::map<unsigned, std::vector<std::pair<char, unsigned> > >::iterator itr =
						uid_to_timestamps.begin(); itr != uid_to_timestamps.end(); itr++) {
			unsigned uid = itr->first;
			all_uids.insert(uid);
			assert (uid_to_rwlogstats.find(uid) != uid_to_rwlogstats.end());
		}


		// Dump to the GZFile as well
		char fn_kind[22], fn_rwsetsize[22], fn_rwset[88];

		sprintf(fn_rwset, "tm_rwsets.gz");

		if (tommy_0729_global_capacity == 0) {
			sprintf(fn_rwsetsize, "infty");
		} else {
			sprintf(fn_rwsetsize, "%u", tommy_0729_global_capacity);
		}

		if (g_tommy_flag_0808 == true) {
			if (g_tommy_flag_0729 == true) {
				sprintf(fn_kind, "08080729");
			} else {
				sprintf(fn_kind, "0808");
			}
			sprintf(fn_rwset, "tm_rwsets_%s_%s.gz", fn_kind, fn_rwsetsize);
		} else {
			if (g_tommy_flag_0729 == true) {
				sprintf(fn_kind, "0729");
				sprintf(fn_rwset, "tm_rwsets_%s_%s.gz", fn_kind, fn_rwsetsize);
			} else {
				sprintf(fn_kind, "orig");
				sprintf(fn_rwset, "tm_rwsets_%s.gz", fn_kind);
			}
		}

		gzFile f_rwsets = gzopen(fn_rwset, "wb");
		for (std::set<unsigned>::iterator itrU = all_uids.begin(); itrU != all_uids.end(); itrU++) {
			unsigned uid = *itrU, epoch = 0;
			num_uid ++;
			std::vector<std::pair<char, unsigned> >* lst1    = &(uid_to_timestamps.at(uid));
			std::vector<TxnEpochStates>* lst2 = &(uid_to_rwlogstats.at(uid));
			char last_event = ' ';
			unsigned last_cycle = 0;
			std::vector<std::pair<char, unsigned> >::iterator    itr1 = lst1->begin();
			std::vector<TxnEpochStates>::iterator itr2 = lst2->begin();

			for (; itr1 != lst1->end() && itr2 != lst2->end(); itr1++) {
				fprintf(x, "%u\t%c\t%u\n", uid, itr1->first, itr1->second);
				num_events ++;

				// Process Events
				char evt = itr1->first;
				if (itr1 != lst1->begin()) {
					const unsigned delta_cycle = itr1->second - last_cycle;
					switch (last_event) {
						case 'S': { sum_exec += delta_cycle; break; }
						case 'A': { sum_exec += delta_cycle; break; }
						case 's': { sum_log_walk += delta_cycle; break; }
						case 'w': { sum_wait_cu += delta_cycle; break; }
						default: break;
					}
				}

				// When we complete an epoch
				if (evt == 'C' || evt == 'A') {
					num_epoches ++;
					TxnEpochStates* rwls = &(*itr2);
					sum_rset_size    += rwls->read_set_size;
					sum_rset_size_nz += rwls->read_set_size_nz;
					sum_wset_size    += rwls->write_set_size;
					sum_wset_size_nz += rwls->write_set_size_nz;
					sum_rset_size_1 += rwls->read_set_size_1;
					sum_wset_size_1 += rwls->write_set_size_1;

					fprintf(f_rwsz, "%u\t%u\t%c\t%u\t%u\t%u\t%u\t%u\t%u\n",
						uid, epoch, evt,
						rwls->read_set_size,  rwls->read_set_size_nz,
						rwls->write_set_size, rwls->write_set_size_nz,
						rwls->read_set_size_1, rwls->write_set_size_1
					);

					itr2 ++;
					epoch ++;

					rwls->appendToGzFile(f_rwsets);
				}

				last_event = evt;
				last_cycle = itr1->second;
			}
		}

		fclose(x);
		fclose(f_rwsz);
		gzclose(f_rwsets);
		printf("Wrote %u UIDs and %u events to tx_all_timeline.txt\n", num_uid, num_events);
		printf("Total & average (across %d epoches) cycles spent in\n", num_epoches);
		printf("   Executing a Txn: %llu, %f\n", sum_exec,    1.0 * sum_exec / num_epoches);
		printf("   in TLW         : %llu, %f\n", sum_log_walk, 1.0 * sum_log_walk / num_epoches);
		printf("   in CU          : %llu, %f\n", sum_wait_cu, 1.0 * sum_wait_cu / num_epoches);
		printf("Total & average (across %d epoches) sizes of RWLogs according to TLW:\n", num_epoches);
		printf("   Read Log            : %llu, %f\n", sum_rset_size, 1.0 * sum_rset_size / num_epoches);
		printf("   Read Log (Nonzero)  : %llu, %f\n", sum_rset_size_nz, 1.0 * sum_rset_size_nz / num_epoches);
		printf("   Write Log           : %llu, %f\n", sum_wset_size, 1.0 * sum_wset_size / num_epoches);
		printf("   Write Log (Nonzero) : %llu, %f\n", sum_wset_size_nz, 1.0 * sum_wset_size_nz / num_epoches);
		printf("Total & average (across %d epoches) sizes of RWLogs according to tm_manager:\n", num_epoches);
		printf("   Read Log            : %llu, %f\n", sum_rset_size_1, 1.0 * sum_rset_size_1 / num_epoches);
		printf("   Write Log           : %llu, %f\n", sum_wset_size_1, 1.0 * sum_wset_size_1 / num_epoches);
	}

	{
		printf("# of TommyPackets = %u, cum. delay = %u, avg delay = %.2f\n",
			num_tommy_packets, cum_tommy_packet_delay, cum_tommy_packet_delay*1.0/num_tommy_packets);
		printf("# of aborted/samewarp/all staggers = %u / %u / %u\n",
			g_num_stagger_aborts, g_num_stagger_samewarp, g_num_staggers);
		if (g_tommy_dbg_0830) {
			printf("# of Sanity Check 0830's: %u\n", g_0830_sanity_check_count);
			printf("Avg # of N-entrys on top of the topmost T entry: %f = %u / %u\n",
					1.0 * g_0830_n_count / g_0830_sanity_check_count, g_0830_n_count, g_0830_sanity_check_count);
			printf("Avg # of T-entrys: %f = %u / %u\n",
					(1.0 * g_0830_t_count) / g_0830_sanity_check_count, g_0830_t_count, g_0830_sanity_check_count);
		}
		printf("# of restoreAll by PC: %u\n", g_0830_restore_by_pc_count);
		printf("# of 32-cycle delays caused by CAT lookup at LD/ST: %ld\n", g_cat_look_delays);
		printf("# of hazards caused by CAT lookup at LD/ST:         %ld\n", g_cat_look_hazards);
		printf("# of ShMem access in intra-warp CD: %u\n", g_iwcd_shmem_count);
		printf("# of LogLoad in intra_warp CD: %u ", g_iwcd_rwlogaccess_count);
		if (g_tommy_flag_1028 > 0) { printf(" (Injected same number of events to model Early Abort)"); }
		printf("# of groups of RCT->CAT packets transfered: %lu\n", g_tommy_packet_count);
		printf("     %lu of them are timeout-induced.\n", g_tommy_packet_count_tout);
		printf("\n");
	}
}

void Cartographer_recordDarkSiliconOpportunity(simd_function_unit* simdfu,
		int sid, int fu_idx, const char* tag) {
	g_cartographer->recordDarkSiliconOpportunity(simdfu, sid, fu_idx, tag);
}
void Cartographer::recordDarkSiliconOpportunity(simd_function_unit* simdfu,
		int sid, int fu_idx, const char* tag) {
	if (!is_log_simdfu_occupancy) return;
	if (simdfu_to_ident.find(simdfu) == simdfu_to_ident.end()) {
		struct FUIdentifier ident;
		ident.fu_idx = fu_idx; ident.sid = sid;
		strcpy(ident.tag, tag);
		simdfu_to_ident[simdfu] = ident;
	}
	if (last_fu_occupancy_state.find(simdfu) == last_fu_occupancy_state.end()) {
		struct FUOccupancyState x;
		for (int i=0; i<32; i++) {
			x.last_update_cycles.push_back(0);
			x.last_update_ticks.push_back(0);
			x.vacant_cycles_histogram.push_back(pow2_histogram());
			x.vacant_ticks_histogram.push_back(pow2_histogram());
			x.vacant_cycles_sum_histogram.push_back(pow2sum_histogram());
			x.vacant_ticks_sum_histogram.push_back(pow2sum_histogram());
			x.active_ticks.push_back(0);
			x.inactive_ticks.push_back(0);
		}
		last_fu_occupancy_state[simdfu] = x;
	}
	struct FUOccupancyState* fuos = &(last_fu_occupancy_state.at(simdfu));
	pipelined_simd_unit* psu = dynamic_cast<pipelined_simd_unit*>(simdfu);
	if (psu) {
		std::bitset<32> occ;
		for (int i=0; i<psu->MAX_ALU_LATENCY; i++) {
			if (psu->occupied.test(i)) {
				warp_inst_t* wi = psu->m_pipeline_reg[i];
				for (int j=0; j<wi->warp_size(); j++) {
					if (wi->active(j)) { occ.set(j); }
				}
			}
		}
		fuos->update(occ);
	}
}

void Cartographer::FUOccupancyState::update(const std::bitset<32>& occupied) {
	curr_tick ++;
	for (int i=0; i<32; i++) {
		if (occupied.test(i)) active_ticks[i] += 1;
		else inactive_ticks[i] += 1;
	}
	if (lane_occupied == occupied) return;
	const unsigned long long curr_cyc = getCurrentCycleCount();

	for (int i=0; i<32; i++) {
		if (lane_occupied.test(i) != occupied.test(i)) {
			if (lane_occupied.test(i) == false) {
				int diff_cyc = curr_cyc - last_update_cycles[i],
					diff_tick = curr_tick - last_update_ticks[i];
				if (diff_cyc > 0) vacant_cycles_histogram.at(i).add2bin(diff_cyc);
				if (diff_tick> 0) vacant_ticks_histogram.at(i).add2bin(diff_tick);
				vacant_cycles_sum_histogram.at(i).add2bin(diff_cyc);
				vacant_ticks_sum_histogram.at(i).add2bin(diff_tick);
			}
			last_update_cycles[i] = curr_cyc;
			last_update_ticks[i]  = curr_tick;
		}
	}
	lane_occupied = occupied;
}

void Cartographer_dumpDarkSiliconOpportunity() {
	g_cartographer->dumpDarkSiliconOpportunity();
}
void Cartographer::dumpDarkSiliconOpportunity() {
	FILE* f_dsso = fopen("dark_silicon_opportunity.txt", "w");
	printf("[Cartographer::dumpDarkSiliconOpportunity]\n");
	for (std::map<simd_function_unit*, struct FUIdentifier>::iterator itr =
		simdfu_to_ident.begin(); itr != simdfu_to_ident.end(); itr++) {
		simd_function_unit* simdfu = itr->first;
		struct FUIdentifier* ident = &(itr->second);
		if (last_fu_occupancy_state.find(simdfu) != last_fu_occupancy_state.end()) {
			struct FUOccupancyState* fuos = &(last_fu_occupancy_state.at(simdfu));

			//               SID IDX Lane Tag ActiveTicks InactiveTicks
			for (int i=0; i<32; i++) {
				fprintf(f_dsso, "%d %d %d %s NA NA Freq ", ident->sid, ident->fu_idx, i, ident->tag);
				fuos->vacant_cycles_histogram.at(i).fprint2(f_dsso);
				fprintf(f_dsso, "\n");
			}
			for (int i=0; i<32; i++) {
				fprintf(f_dsso, "%d %d %d %s %llu %llu WeightSum ", ident->sid, ident->fu_idx, i, ident->tag,
					fuos->active_ticks.at(i), fuos->inactive_ticks.at(i));
				fuos->vacant_cycles_sum_histogram.at(i).fprint2(f_dsso);
				fprintf(f_dsso, "\n");
			}
		} else {
			printf("SID[%d] %s[%d] No LastFUOccupancyState !?\n",
				ident->sid, ident->tag, ident->fu_idx);
		}
	}
	fclose(f_dsso);
}

// --- Default link map for the INTEGRITY Kernel Address Space ---
//
// Various sections are described in the comment block at the end of this file.
//
// More information can be found in the "Linker Directives Files" section of
// the "Building INTEGRITY Applications" chapter of the INTEGRITY Development
// Guide, and in the "Linker Directives Files" section of the "File Structure
// chapter of the INTEGRITY BSP User's Guide.
//

CONSTANTS
{
        __INTEGRITY_DebugBufferSize      =    0x10000
        __INTEGRITY_HeapSize             =    0x60000
        __INTEGRITY_StackSize            =     0x4000
        __INTEGRITY_DownloadSize         =   0x180000
        __INTEGRITY_MaxCoreSize          =   0x200000
        __INTEGRITY_FirstKernelAddress   = 0x00000000
        __INTEGRITY_RamLimit             = 0x00800000
        __INTEGRITY_RamLimitSize         =          0
        __INTEGRITY_SimFlashSize         =   0x200000
        __INTEGRITY_InMemTMSize          =    0x40000
}


SECTIONS
{
	.vector	  	0x00000000			:
	.ramstart 	0x4000				:
	.PPC.EMB.sdata0					:
	.PPC.EMB.sbss0	CLEAR				:
	.text						:
        .syscall					:
 	.fixaddr					:
	.fixtype					:
        .secinfo					:
	.robase						:
	.execname					:
	.sdata2		align(16)			:
	.rodata		align(16)			:
        .simdata                                        :
	.initrodata					:
	.sdabase	align(8)			:
	.sdata		align(8)			:
        .sbss						:
        .data						:
	.dbsockemu	align(4)			:
        .bss						:
  	.download	NOCLEAR	: { . += isdefined(__INTEGRITY_LoaderTask) ? __INTEGRITY_DownloadSize : 0; }
	.mr_rwe__download  align(0x1000)	NOCLEAR	: { . += isdefined(VirtualLoaderHelperInit) ? __INTEGRITY_DownloadSize : 0; }
	.debugbuffer	NOCLEAR : { . += isdefined(_logchar) ? __INTEGRITY_DebugBufferSize : 0; }
	.eventlog	align(4)   NOCLEAR	: { . += isdefined(MEV_Init) ? 0x20000 : 0; }
	.heap		align(16) pad(__INTEGRITY_HeapSize)	NOCLEAR	:
	.stack		align(16) pad(__INTEGRITY_StackSize)	NOCLEAR :
	.kstack		align(16) pad(0x4000)	NOCLEAR :
        .mr___flashbuf  align(4)  pad(0x20000)  NOCLEAR :
	.mr__core align(0x1000) NOCLEAR : { . += isdefined(DumpKernelCore) ? (min(__INTEGRITY_MaxCoreSize,(sizeof(.kstack)+sizeof(.bss)+sizeof(.sbss)+sizeof(.data)+sizeof(.sdata)+0x4000))) : 0; }
	.mr_rwe_inmemtm align(0x1000) NOCLEAR : { . += isdefined(InMemTM) ? __INTEGRITY_InMemTMSize : 0; } __ghsbegin_inmemtm = addr(.mr_rwe_inmemtm); __ghsend_inmemtm = addr(.mr_rwe_inmemtm) + sizeof(.mr_rwe_inmemtm);
	.indrtVersion					: { . += isdefined(Process_Debugging_Using_INDRT2) ? 2: 1;}
	.ramend	 	align(0x1000)			:
	.ramlimit 	__INTEGRITY_RamLimitSize==0?__INTEGRITY_RamLimit:__INTEGRITY_FirstKernelAddress+__INTEGRITY_RamLimitSize		:

        .altromstart      0x10000000                    :
        .flashdevchipbase                               :
        .flashdevstart    align(0x2000)                 :
        .flashdevend      ADDR(.altromstart)+__INTEGRITY_SimFlashSize :
        .altromend                                      :

	.romstart	0xfc000000			:
	.romend						:
}


//
// Section Notes:
//	.ramstart, .ramend,  .romstart,  .romend are zero-size sections
//	marking the beginning and end of the RAM and FLASH/ROM areas used
//	by the KernelSpace program (the kernel, drivers, INDRT, etc.).
// 	This section map is appropriate only for a RAM resident application
//	(see flash.ld for a flashable setup).  Memory addresses before .ramend
//	are used for the KernelSpace program.  Memory addresses between .ramend
//	and .ramlimit are used for the memory mappings of virtual AddressSpaces
//	as well as all other kernel-required memory (i.e. for Objects and
//	pages). To increase the amount of simulated RAM, increase the address
//	of .ramlimit.  No program sections are within the region defined by
//	.romstart and .romend, although other non-program section may appear
//	within the region. Even if the region between .romstart and .romend is
//	empty, the sections must still be defined in the link map because of
//	symbol references.
//
//	sdabase and sdata2 need to be aligned to sdata and rodata,
//	respectively.
//
//	.initrodata is constant data used specifically during BSP memory 
//	initialization.	.rodata is the regular const data used by the
//	kernel, libraries, and user application.
//
//	.kstack represents the area of stack used by the kernel and its
//	interrupt handlers.     NOCLEAR must be present,  it keeps
//	GHS downloaders from zeroing out this memory.
//
//	.stack represents the area of stack used by the Initial Task.
//	If the user links a main() routine with the kernel,  main() is the
//	entry point for Initial Task.    
//
//	.heap represents dynamically allocated memory for the application.
//
//	.execname is a special section filled in by the GHS linker; it contains
//	the name of the image resulting from this link.    It is used by
//	the kernel to inform the debugger of the executable file that should
//	be used when a task from this application is attached.
//
//	.vector contains the machine vectors for interrupt handlers.
//	It starts at 0 in simulation mode.
//
//    .mr__core is used to hold core dumps.  It is automatically 0 sized
//    if libcore.a is not linked in
//
//	.download is used to hold dynamic download images in memory temporarily
//	before they are processed by the bootloader.    The size of the
//	.download section must be at least as big as any image that will be
//	downloaded dynamically.   If an image will not fit,  .download
//	can be increased (and the KernelSpace program rebuilt),  although
//	be careful not to exceed the board's RAM requirements.
//
//	.eventlog is for MULTI EventAnalyzer support.   This buffer
//	is used to store the event log on the target (includes
//	accounting information as well as the event buffer itself).
//	Making the size of this section larger enables more data to
//	be logged during Postmortem Mode (circular buffer is used)
//	and also may reduce the network traffic during Live Mode.
//		
//	The other sections are standard sections for a link map.   
//	This map can be customized by the user,  for example to make
//	the .heap or .stack larger,  add user-defined sections, etc.
//	
//

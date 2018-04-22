// --- Flash-based link map for the INTEGRITY Kernel Address Space ---
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
        __INTEGRITY_DebugBufferSize   =  0x10000
        __INTEGRITY_HeapSize          =  0x60000
        __INTEGRITY_StackSize         =   0x4000
        __INTEGRITY_DownloadSize      =  0x60000
        __INTEGRITY_MaxCoreSize       = 0x200000
        __INTEGRITY_SimFlashSize      = 0x200000
}


SECTIONS
{
	.vector		0x00000000			:
	.ramstart	0x4000				:
        .syscall					:
	.robase						:
	.initrodata					:
	.sdabase	align(8)			:
	.sdata		align(8)			:
        .sbss						:
        .data						:
	.dbsockemu	align(4)			:
        .bss						:
	.fixaddr					:
	.fixtype					:
        .secinfo					:	
  	.download	NOCLEAR	: { . += isdefined(__INTEGRITY_LoaderTask) ? __INTEGRITY_DownloadSize : 0; }
  	.mr_rwe__download  align(0x1000)	NOCLEAR	: { . += isdefined(VirtualLoaderHelperInit) ? __INTEGRITY_DownloadSize : 0; }
	.debugbuffer	NOCLEAR : { . += isdefined(_logchar) ? __INTEGRITY_DebugBufferSize : 0; }
	.eventlog	align(4)   NOCLEAR	: { . += isdefined(MEV_Init) ? 0x20000 : 0; }
	.heap		align(16) pad(__INTEGRITY_HeapSize)	NOCLEAR	:
	.stack		align(16) pad(__INTEGRITY_StackSize)	NOCLEAR :
	.kstack		align(16) pad(0x4000)	NOCLEAR :
        .mr___flashbuf  align(4)  pad(0x20000)  NOCLEAR :
	.mr__core align(0x1000) NOCLEAR : { . += isdefined(DumpKernelCore) ? (min(__INTEGRITY_MaxCoreSize,(sizeof(.kstack)+sizeof(.bss)+sizeof(.sbss)+sizeof(.data)+sizeof(.sdata)+0x4000))) : 0; }
	.ramend		align(0x1000)			:
	.ramlimit 	0x200000 			:

        .altromstart      0x10000000                    :
        .flashdevchipbase                               :
        .flashdevstart    align(0x2000)                 :
        .flashdevend      ADDR(.altromstart)+__INTEGRITY_SimFlashSize :
        .altromend                                      :

        .romstart	0xfe000000			:
	.rtext						:
	.PPC.EMB.sdata0					:
	.PPC.EMB.sbss0	CLEAR				:
	.text 						:
	.sdata2	align(16)				:
	.rrodata align(16) 				:
	.rodata align(16)				:
        .simdata                                        :
	.rexecname	 				:
	.execname					:
	.rsyscall	ROM(.syscall)			:
	.romsdata  	ROM(.sdata)			:
	.rdata 		ROM(.data)  			:
	.rdbsockemu	ROM(.dbsockemu)			:
	.romend						:
}


//
// Section Notes:
//	See default.ld for a section map appropriate for fully RAM-resident
//	applications.    This flash.ld file is appropriate for flashable
//	applications.    Also consult default.ld for a description of
//	other special sections that are not described here.
//
//	.ramstart is a zero-size section marking the start of RAM area used
//	by the KernelSpace program.
//
//	.ramend must match the value of "MinimumAddress" specified in
//	the BSP Description File (e.g. default.bsp) used for any applications
//	that are built to run on this board.    Memory addresses before
//	.ramend are used for the KernelSpace program.  Memory addresses between
//	.ramend and .ramlimit are used for the memory mappings of virtual
//	AddressSpaces as well as all other kernel-required memory (i.e. for
//	Objects and pages). To increase the amount of simulated RAM, increase
//	the address of .ramlimit.  If a different ratio of RAM is desired,
//	.ramend and "MinimumAddress" can be changed,  but they both must be
//	changed to the same value.    If too much code/data is added to the
//	KernelSpace program and the address specified by .ramend is
//	exceeded,  the linker will detect an overlap with .ramend and produce 
//	an error.   
//
//	sdabase and sdata2 need to be aligned to sdata and rodata,
//	respectively.
//
//	.mr___flashbuf is used to hold a flash sector when using the 'nc'
//	command to save a network configuration to flash.   The sector
//	size on the Am29F040 flash chip on the ADS board is 256K bytes.
//	Once the flash is configured,  if the 256K of RAM needs to be 
//	recovered for user code linker in KernelSpace, change the section as 
//	follows:
//
//	    .mr___flashbuf					:
//	
//	and then relink the kernel image.  The kernel automatically detects 
//	that the flashbuf is zero size and will not allow the 'ncs' command 
//	to save.    The section can not be removed from the link map.
//	See the INTEGRITY Development Guide for details on the 'nc' commands.
//
//       .mr__core is used to hold core dumps.  It is automatically 0 sized
//       if libcore.a is not linked in
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
//	.romstart is a zero-size section marking the start of the FLASH/ROM 
//	area used by the application.    The .vector section actually 
//	precedes this in ROM,  but it contains an image of the exception
//	vector handlers only,  not part of the actual application.   
//	.romend is a zero-size section marking the end of the FLASH/ROM area
//	used by the application code and data.     .romend is fixed for this
//	BSP and must match the value of MinimumROMAddress specified in the
//	default.bsp file.    Addresses after .romend/MinimumROMAddress are
//	used by the Integrate utility for virtual AddressSpace code and data
//	that must be copied from ROM to RAM.   If the .romend address is
//	increased,  the default.bsp MinimumROMAddress must be changed 
//	accordingly.
//
//	In this link map,  the RAM and ROM areas are single, contiguous areas
//	of memory.   The proper operation of the kernel for this BSP depends
//	on this being the case.
//
//	This file is linked so that the kernel image will both boot and
//	run out of ROM (i.e. there is no ROM to RAM copy of the kernel
//	image).  This is caused by .text directly after .rtext.  This
//	is done with .rodata/.rrodata and .execname/.rexecname as well.  In order to 
//	boot out of ROM and run of of RAM, simply move the .text 
//	declaration directly after .ramstart and add ROM(.text)
//	after .rtext (and similarly for .rodata and .execname)  For .rodata, also
//	move .sdata2 before .rodata, so the PICBASE is calculated correctly.  Be sure
//	to keep the sections in the same relative order and do not add any other
//	sections between them.
//
//	IMPORTANT!!  The .text section must start at the same offset from
//	.ramstart as the .rtext section (ROM image of .text) starts from
//	.romstart.    The kernel relies on this when it jumps to ROM code
//	from the initial reset vector;  the relative offset of _start in
//	the RAM area is added to the .romstart location to find the ROM image
//	of _start.   The easiest way to ensure correct operation is to place
//	.text directly after .ramstart and .rtext directly after .romstart.
//
//

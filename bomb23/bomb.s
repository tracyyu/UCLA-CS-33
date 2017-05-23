	.file	"bomb.c"
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"%s: Error: Couldn't open %s\n"
.LC2:
	.string	"Usage: %s [<input_file>]\n"
	.align 8
.LC3:
	.string	"Welcome to my fiendish little bomb. You have 6 phases with"
	.align 8
.LC4:
	.string	"which to blow yourself up. Have a nice day!"
	.align 8
.LC5:
	.string	"Phase 1 defused. How about the next one?"
.LC6:
	.string	"That's number 2.  Keep going!"
.LC7:
	.string	"Halfway there!"
	.align 8
.LC8:
	.string	"So you got that one.  Try this one."
.LC9:
	.string	"Good work!  On to the next..."
	.text
.globl main
	.type	main, @function
main:
.LFB5:
	pushq	%rbp
.LCFI0:
	movq	%rsp, %rbp
.LCFI1:
	subq	$32, %rsp
.LCFI2:
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	cmpl	$1, -20(%rbp)
	jne	.L2
	movq	stdin(%rip), %rax
	movq	%rax, infile(%rip)
	jmp	.L4
.L2:
	cmpl	$2, -20(%rbp)
	jne	.L5
	movq	-32(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdi
	movl	$.LC0, %esi
	call	fopen
	movq	%rax, infile(%rip)
	movq	infile(%rip), %rax
	testq	%rax, %rax
	jne	.L4
	movq	-32(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	movq	-32(%rbp), %rax
	movq	(%rax), %rsi
	movl	$.LC1, %edi
	movl	$0, %eax
	call	printf
	movl	$8, %edi
	call	exit
.L5:
	movq	-32(%rbp), %rax
	movq	(%rax), %rsi
	movl	$.LC2, %edi
	movl	$0, %eax
	call	printf
	movl	$8, %edi
	call	exit
.L4:
	movl	$0, %eax
	call	initialize_bomb
	movl	$.LC3, %edi
	call	puts
	movl	$.LC4, %edi
	call	puts
	movl	$0, %eax
	call	read_line
	cltq
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdi
	movl	$0, %eax
	call	phase_1
	movl	$0, %eax
	call	phase_defused
	movl	$.LC5, %edi
	call	puts
	movl	$0, %eax
	call	read_line
	cltq
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdi
	movl	$0, %eax
	call	phase_2
	movl	$0, %eax
	call	phase_defused
	movl	$.LC6, %edi
	call	puts
	movl	$0, %eax
	call	read_line
	cltq
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdi
	movl	$0, %eax
	call	phase_3
	movl	$0, %eax
	call	phase_defused
	movl	$.LC7, %edi
	call	puts
	movl	$0, %eax
	call	read_line
	cltq
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdi
	movl	$0, %eax
	call	phase_4
	movl	$0, %eax
	call	phase_defused
	movl	$.LC8, %edi
	call	puts
	movl	$0, %eax
	call	read_line
	cltq
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdi
	movl	$0, %eax
	call	phase_5
	movl	$0, %eax
	call	phase_defused
	movl	$.LC9, %edi
	call	puts
	movl	$0, %eax
	call	read_line
	cltq
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdi
	movl	$0, %eax
	call	phase_6
	movl	$0, %eax
	call	phase_defused
	movl	$0, %eax
	leave
	ret
.LFE5:
	.size	main, .-main
	.section	.eh_frame,"a",@progbits
.Lframe1:
	.long	.LECIE1-.LSCIE1
.LSCIE1:
	.long	0x0
	.byte	0x1
	.string	"zR"
	.uleb128 0x1
	.sleb128 -8
	.byte	0x10
	.uleb128 0x1
	.byte	0x3
	.byte	0xc
	.uleb128 0x7
	.uleb128 0x8
	.byte	0x90
	.uleb128 0x1
	.align 8
.LECIE1:
.LSFDE1:
	.long	.LEFDE1-.LASFDE1
.LASFDE1:
	.long	.LASFDE1-.Lframe1
	.long	.LFB5
	.long	.LFE5-.LFB5
	.uleb128 0x0
	.byte	0x4
	.long	.LCFI0-.LFB5
	.byte	0xe
	.uleb128 0x10
	.byte	0x86
	.uleb128 0x2
	.byte	0x4
	.long	.LCFI1-.LCFI0
	.byte	0xd
	.uleb128 0x6
	.align 8
.LEFDE1:
	.ident	"GCC: (GNU) 4.1.2 20080704 (Red Hat 4.1.2-52)"
	.section	.note.GNU-stack,"",@progbits

	.file	"sfmult_atxtyt_k.c"
	.text
.globl sfmult_AT_XT_YT_2
	.type	sfmult_AT_XT_YT_2, @function
sfmult_AT_XT_YT_2:
.LFB29:
	pushl	%ebp
.LCFI0:
	movl	%esp, %ebp
.LCFI1:
	pushl	%edi
.LCFI2:
	pushl	%esi
.LCFI3:
	pushl	%ebx
.LCFI4:
	subl	$20, %esp
.LCFI5:
	movl	40(%ebp), %esi
	movl	36(%ebp), %eax
	testl	%eax, %eax
	jle	.L13
	movl	60(%ebp), %eax
	sall	$3, %eax
	movl	%eax, -16(%ebp)
	movl	8(%ebp), %eax
	movl	%eax, -20(%ebp)
	xorl	%edi, %edi
	movl	$0, -24(%ebp)
	pxor	%xmm6, %xmm6
	movl	-24(%ebp), %edx
.L4:
	movl	16(%ebp), %eax
	movl	4(%eax,%edx,4), %edx
	movl	%edx, -28(%ebp)
	subl	%edi, %edx
	movl	%edx, -32(%ebp)
	movl	$1431655766, %edx
	movl	-32(%ebp), %eax
	imull	%edx
	movl	-32(%ebp), %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	leal	(%edx,%edx,2), %edx
	movl	-32(%ebp), %eax
	subl	%edx, %eax
	cmpl	$1, %eax
	je	.L6
	cmpl	$2, %eax
	je	.L7
	movapd	%xmm6, %xmm4
	movapd	%xmm6, %xmm5
.L8:
	cmpl	%edi, -28(%ebp)
	jle	.L10
	movl	20(%ebp), %eax
	leal	(%eax,%edi,4), %ecx
	movl	24(%ebp), %eax
	leal	(%eax,%edi,8), %edx
.L12:
	movsd	(%edx), %xmm1
	movsd	8(%edx), %xmm2
	movsd	16(%edx), %xmm3
	movl	(%ecx), %eax
	sall	$4, %eax
	movapd	%xmm1, %xmm0
	mulsd	(%eax,%esi), %xmm0
	addsd	%xmm0, %xmm5
	mulsd	8(%esi,%eax), %xmm1
	addsd	%xmm1, %xmm4
	movl	4(%ecx), %eax
	sall	$4, %eax
	movapd	%xmm2, %xmm0
	mulsd	(%esi,%eax), %xmm0
	addsd	%xmm0, %xmm5
	mulsd	8(%esi,%eax), %xmm2
	addsd	%xmm2, %xmm4
	movl	8(%ecx), %eax
	sall	$4, %eax
	movapd	%xmm3, %xmm0
	mulsd	(%esi,%eax), %xmm0
	addsd	%xmm0, %xmm5
	mulsd	8(%esi,%eax), %xmm3
	addsd	%xmm3, %xmm4
	addl	$3, %edi
	addl	$12, %ecx
	addl	$24, %edx
	cmpl	%edi, -28(%ebp)
	jg	.L12
.L10:
	movl	-20(%ebp), %edx
	movsd	%xmm5, (%edx)
	movsd	%xmm4, 8(%edx)
	addl	$1, -24(%ebp)
	movl	-16(%ebp), %eax
	addl	%eax, %edx
	movl	%edx, -20(%ebp)
	movl	-24(%ebp), %edx
	cmpl	%edx, 36(%ebp)
	jne	.L4
.L13:
	addl	$20, %esp
	popl	%ebx
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
.L6:
	movapd	%xmm6, %xmm1
	movapd	%xmm6, %xmm2
	movl	20(%ebp), %edx
.L9:
	movl	24(%ebp), %eax
	movsd	(%eax,%edi,8), %xmm0
	movl	(%edx,%edi,4), %eax
	sall	$4, %eax
	movapd	%xmm0, %xmm5
	mulsd	(%esi,%eax), %xmm5
	addsd	%xmm2, %xmm5
	movapd	%xmm0, %xmm4
	mulsd	8(%esi,%eax), %xmm4
	addsd	%xmm1, %xmm4
	addl	$1, %edi
	jmp	.L8
.L7:
	movl	24(%ebp), %edx
	movsd	(%edx,%edi,8), %xmm0
	movl	20(%ebp), %edx
	movl	(%edx,%edi,4), %eax
	sall	$4, %eax
	movapd	%xmm0, %xmm2
	mulsd	(%esi,%eax), %xmm2
	addsd	%xmm6, %xmm2
	movapd	%xmm0, %xmm1
	mulsd	8(%esi,%eax), %xmm1
	addsd	%xmm6, %xmm1
	addl	$1, %edi
	jmp	.L9
.LFE29:
	.size	sfmult_AT_XT_YT_2, .-sfmult_AT_XT_YT_2
.globl sfmult_AT_XT_YT_3
	.type	sfmult_AT_XT_YT_3, @function
sfmult_AT_XT_YT_3:
.LFB30:
	pushl	%ebp
.LCFI6:
	movl	%esp, %ebp
.LCFI7:
	pushl	%edi
.LCFI8:
	pushl	%esi
.LCFI9:
	pushl	%ebx
.LCFI10:
	subl	$16, %esp
.LCFI11:
	movl	40(%ebp), %edi
	movl	36(%ebp), %edx
	testl	%edx, %edx
	jle	.L27
	movl	60(%ebp), %eax
	sall	$3, %eax
	movl	%eax, -16(%ebp)
	movl	8(%ebp), %eax
	movl	%eax, -20(%ebp)
	xorl	%esi, %esi
	movl	$0, -24(%ebp)
	pxor	%xmm6, %xmm6
	movl	-24(%ebp), %edx
.L20:
	movl	16(%ebp), %eax
	movl	4(%eax,%edx,4), %edx
	movl	%edx, -28(%ebp)
	movl	%edx, %eax
	subl	%esi, %eax
	testb	$1, %al
	jne	.L21
	movapd	%xmm6, %xmm3
	movapd	%xmm6, %xmm4
	movapd	%xmm6, %xmm5
.L23:
	cmpl	%esi, -28(%ebp)
	jle	.L24
	movl	20(%ebp), %eax
	leal	(%eax,%esi,4), %ecx
	movl	24(%ebp), %eax
	leal	(%eax,%esi,8), %edx
.L26:
	movsd	(%edx), %xmm1
	movsd	8(%edx), %xmm2
	movl	(%ecx), %eax
	sall	$5, %eax
	movapd	%xmm1, %xmm0
	mulsd	(%eax,%edi), %xmm0
	addsd	%xmm0, %xmm5
	leal	(%edi,%eax), %eax
	movapd	%xmm1, %xmm0
	mulsd	8(%eax), %xmm0
	addsd	%xmm0, %xmm4
	mulsd	16(%eax), %xmm1
	addsd	%xmm1, %xmm3
	movl	4(%ecx), %eax
	sall	$5, %eax
	movapd	%xmm2, %xmm0
	mulsd	(%edi,%eax), %xmm0
	addsd	%xmm0, %xmm5
	leal	(%edi,%eax), %eax
	movapd	%xmm2, %xmm0
	mulsd	8(%eax), %xmm0
	addsd	%xmm0, %xmm4
	mulsd	16(%eax), %xmm2
	addsd	%xmm2, %xmm3
	addl	$2, %esi
	addl	$8, %ecx
	addl	$16, %edx
	cmpl	%esi, -28(%ebp)
	jg	.L26
.L24:
	movl	-20(%ebp), %edx
	movsd	%xmm5, (%edx)
	movsd	%xmm4, 8(%edx)
	movsd	%xmm3, 16(%edx)
	addl	$1, -24(%ebp)
	movl	-16(%ebp), %eax
	addl	%eax, %edx
	movl	%edx, -20(%ebp)
	movl	-24(%ebp), %edx
	cmpl	%edx, 36(%ebp)
	jne	.L20
.L27:
	addl	$16, %esp
	popl	%ebx
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
.L21:
	movl	24(%ebp), %eax
	movsd	(%eax,%esi,8), %xmm0
	movl	20(%ebp), %edx
	movl	(%edx,%esi,4), %eax
	sall	$5, %eax
	movapd	%xmm0, %xmm5
	mulsd	(%edi,%eax), %xmm5
	addsd	%xmm6, %xmm5
	leal	(%edi,%eax), %eax
	movapd	%xmm0, %xmm4
	mulsd	8(%eax), %xmm4
	addsd	%xmm6, %xmm4
	movapd	%xmm0, %xmm3
	mulsd	16(%eax), %xmm3
	addsd	%xmm6, %xmm3
	addl	$1, %esi
	jmp	.L23
.LFE30:
	.size	sfmult_AT_XT_YT_3, .-sfmult_AT_XT_YT_3
.globl sfmult_AT_XT_YT_4
	.type	sfmult_AT_XT_YT_4, @function
sfmult_AT_XT_YT_4:
.LFB31:
	pushl	%ebp
.LCFI12:
	movl	%esp, %ebp
.LCFI13:
	pushl	%edi
.LCFI14:
	pushl	%esi
.LCFI15:
	pushl	%ebx
.LCFI16:
	subl	$20, %esp
.LCFI17:
	movl	36(%ebp), %ecx
	testl	%ecx, %ecx
	jle	.L39
	movl	60(%ebp), %eax
	sall	$3, %eax
	movl	%eax, -16(%ebp)
	movl	8(%ebp), %eax
	movl	%eax, -20(%ebp)
	movl	$0, -28(%ebp)
	movl	$0, -24(%ebp)
	pxor	%xmm6, %xmm6
.L33:
	movl	-24(%ebp), %edx
	movl	16(%ebp), %ecx
	movl	4(%ecx,%edx,4), %eax
	cmpl	-28(%ebp), %eax
	jle	.L43
	movl	-28(%ebp), %esi
	movl	20(%ebp), %edi
	leal	(%edi,%esi,4), %ecx
	movl	24(%ebp), %edi
	leal	(%edi,%esi,8), %edx
	movapd	%xmm6, %xmm2
	movapd	%xmm6, %xmm5
	movapd	%xmm6, %xmm4
	movapd	%xmm6, %xmm3
	xorl	%esi, %esi
	subl	-28(%ebp), %eax
	movl	%eax, -32(%ebp)
	movl	40(%ebp), %edi
.L37:
	movsd	(%edx), %xmm1
	movl	(%ecx), %eax
	sall	$5, %eax
	movapd	%xmm1, %xmm0
	mulsd	(%eax,%edi), %xmm0
	addsd	%xmm0, %xmm3
	addl	%edi, %eax
	movapd	%xmm1, %xmm0
	mulsd	8(%eax), %xmm0
	addsd	%xmm0, %xmm4
	movapd	%xmm1, %xmm0
	mulsd	16(%eax), %xmm0
	addsd	%xmm0, %xmm5
	mulsd	24(%eax), %xmm1
	addsd	%xmm1, %xmm2
	addl	$1, %esi
	addl	$4, %ecx
	addl	$8, %edx
	cmpl	-32(%ebp), %esi
	jne	.L37
	addl	%esi, -28(%ebp)
.L36:
	movl	-20(%ebp), %eax
	movsd	%xmm3, (%eax)
	movsd	%xmm4, 8(%eax)
	movsd	%xmm5, 16(%eax)
	movsd	%xmm2, 24(%eax)
	addl	$1, -24(%ebp)
	movl	-16(%ebp), %edx
	addl	%edx, %eax
	movl	%eax, -20(%ebp)
	movl	-24(%ebp), %ecx
	cmpl	%ecx, 36(%ebp)
	jne	.L33
.L39:
	addl	$20, %esp
	popl	%ebx
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
.L43:
	movapd	%xmm6, %xmm2
	movapd	%xmm6, %xmm5
	movapd	%xmm6, %xmm4
	movapd	%xmm6, %xmm3
	jmp	.L36
.LFE31:
	.size	sfmult_AT_XT_YT_4, .-sfmult_AT_XT_YT_4
.globl sfmult_AT_XT_YT_k
	.type	sfmult_AT_XT_YT_k, @function
sfmult_AT_XT_YT_k:
.LFB32:
	pushl	%ebp
.LCFI18:
	movl	%esp, %ebp
.LCFI19:
	pushl	%edi
.LCFI20:
	pushl	%esi
.LCFI21:
	pushl	%ebx
.LCFI22:
	subl	$36, %esp
.LCFI23:
	movl	60(%ebp), %esi
	movl	36(%ebp), %edi
	testl	%edi, %edi
	jle	.L64
	movl	%esi, %edi
	andl	$-2147483645, %edi
	js	.L69
.L47:
	leal	0(,%esi,8), %eax
	movl	%eax, -44(%ebp)
	leal	0(,%edi,8), %edx
	movl	%edx, -16(%ebp)
	movl	8(%ebp), %ecx
	addl	%edx, %ecx
	movl	%ecx, -20(%ebp)
	movl	$0, -40(%ebp)
	movl	$0, -36(%ebp)
.L48:
	movl	-36(%ebp), %eax
	movl	16(%ebp), %ecx
	movl	4(%ecx,%eax,4), %edx
	testl	%esi, %esi
	jle	.L49
	xorl	%eax, %eax
	movl	8(%ebp), %ecx
.L51:
	movl	$0, (%ecx,%eax,8)
	movl	$0, 4(%ecx,%eax,8)
	addl	$1, %eax
	cmpl	%eax, %esi
	jne	.L51
.L49:
	cmpl	-40(%ebp), %edx
	jle	.L52
	movl	-40(%ebp), %eax
	movl	20(%ebp), %ecx
	leal	(%ecx,%eax,4), %eax
	movl	%eax, -28(%ebp)
	movl	-40(%ebp), %ecx
	movl	24(%ebp), %eax
	leal	(%eax,%ecx,8), %ecx
	movl	%ecx, -24(%ebp)
	movl	$0, -32(%ebp)
	subl	-40(%ebp), %edx
	movl	%edx, -48(%ebp)
.L54:
	movl	-24(%ebp), %eax
	movsd	(%eax), %xmm1
	movl	%esi, %eax
	movl	-28(%ebp), %edx
	imull	(%edx), %eax
	movl	40(%ebp), %ecx
	leal	(%ecx,%eax,8), %edx
	cmpl	$2, %edi
	je	.L57
	cmpl	$3, %edi
	je	.L58
	cmpl	$1, %edi
	je	.L70
.L55:
	cmpl	%edi, %esi
	.p2align 4,,5
	jle	.L59
	movl	-20(%ebp), %eax
	addl	-16(%ebp), %edx
	movl	%edi, %ecx
.L61:
	movapd	%xmm1, %xmm0
	mulsd	(%edx), %xmm0
	addsd	(%eax), %xmm0
	movsd	%xmm0, (%eax)
	movapd	%xmm1, %xmm0
	mulsd	8(%edx), %xmm0
	addsd	8(%eax), %xmm0
	movsd	%xmm0, 8(%eax)
	movapd	%xmm1, %xmm0
	mulsd	16(%edx), %xmm0
	addsd	16(%eax), %xmm0
	movsd	%xmm0, 16(%eax)
	movapd	%xmm1, %xmm0
	mulsd	24(%edx), %xmm0
	addsd	24(%eax), %xmm0
	movsd	%xmm0, 24(%eax)
	addl	$4, %ecx
	addl	$32, %eax
	addl	$32, %edx
	cmpl	%ecx, %esi
	jg	.L61
.L59:
	addl	$1, -32(%ebp)
	addl	$4, -28(%ebp)
	addl	$8, -24(%ebp)
	movl	-48(%ebp), %edx
	cmpl	%edx, -32(%ebp)
	jne	.L54
	movl	-32(%ebp), %ecx
	addl	%ecx, -40(%ebp)
.L52:
	addl	$1, -36(%ebp)
	movl	-44(%ebp), %eax
	addl	%eax, -20(%ebp)
	movl	-36(%ebp), %edx
	cmpl	%edx, 36(%ebp)
	je	.L64
	addl	%eax, 8(%ebp)
	jmp	.L48
.L58:
	movapd	%xmm1, %xmm0
	mulsd	16(%edx), %xmm0
	movl	8(%ebp), %eax
	addsd	16(%eax), %xmm0
	movsd	%xmm0, 16(%eax)
.L57:
	movapd	%xmm1, %xmm0
	mulsd	8(%edx), %xmm0
	movl	8(%ebp), %ecx
	addsd	8(%ecx), %xmm0
	movsd	%xmm0, 8(%ecx)
	movl	%ecx, %eax
.L56:
	movapd	%xmm1, %xmm0
	mulsd	(%edx), %xmm0
	addsd	(%eax), %xmm0
	movsd	%xmm0, (%eax)
	jmp	.L55
.L70:
	movl	8(%ebp), %eax
	jmp	.L56
.L64:
	addl	$36, %esp
	popl	%ebx
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
.L69:
	subl	$1, %edi
	orl	$-4, %edi
	addl	$1, %edi
	jmp	.L47
.LFE32:
	.size	sfmult_AT_XT_YT_k, .-sfmult_AT_XT_YT_k
	.ident	"GCC: (GNU) 4.1.0 (SUSE Linux)"
	.section	.note.GNU-stack,"",@progbits

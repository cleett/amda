//! Recursive-descent parser for NVIDIA PTX modules, `.func` helpers, and `.entry` kernels.

use crate::error::CompileError;
use crate::ptx::ast::{
    PtxFunc, PtxInstruction, PtxKernel, PtxLocalDecl, PtxModule, PtxOperand, PtxParam, PtxPredicate,
    PtxRegDecl, PtxSharedDecl, PtxStatement, PtxType,
};
use crate::ptx::lexer::{tokenize, Span, Token, TokenKind};

/// Parses PTX assembly source text into a [`PtxModule`].
pub fn parse_ptx(src: &str) -> Result<PtxModule, CompileError> {
    let tokens = tokenize(src)?;
    let mut parser = Parser { tokens, pos: 0 };
    parser.parse_module()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .unwrap_or_else(|| self.tokens.last().expect("lexer always emits Eof"))
    }

    fn peek_next(&self) -> Option<&Token> {
        self.tokens.get(self.pos + 1)
    }

    fn advance(&mut self) -> Token {
        let tok = self.peek().clone();
        if !matches!(tok.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        tok
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn err<T>(&self, span: Span, msg: impl Into<String>) -> Result<T, CompileError> {
        Err(CompileError::ParseError {
            line: span.line,
            col: span.col,
            message: msg.into(),
        })
    }

    fn expect_ident(&mut self) -> Result<String, CompileError> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Ident(s) => Ok(s),
            _ => self.err(tok.span, format!("expected identifier, got {:?}", tok.kind)),
        }
    }

    fn expect_u32(&mut self) -> Result<u32, CompileError> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Int(v) => u32::try_from(v).map_err(|_| CompileError::ParseError {
                line: tok.span.line,
                col: tok.span.col,
                message: "integer out of u32 range".to_owned(),
            }),
            _ => self.err(tok.span, format!("expected integer, got {:?}", tok.kind)),
        }
    }

    fn expect_kind(
        &mut self,
        check: impl Fn(&TokenKind) -> bool,
        expected: &str,
    ) -> Result<Token, CompileError> {
        let tok = self.advance();
        if check(&tok.kind) {
            Ok(tok)
        } else {
            self.err(tok.span, format!("expected {expected}, got {:?}", tok.kind))
        }
    }

    fn parse_module(&mut self) -> Result<PtxModule, CompileError> {
        let mut version = "7.0".to_owned();
        let mut target = "sm_80".to_owned();
        let mut address_size = 64u32;
        let mut shared_decls = Vec::new();
        let mut funcs = Vec::new();
        let mut kernels = Vec::new();

        while !self.at_eof() {
            let tok = self.peek().clone();
            match &tok.kind {
                TokenKind::Directive(d) => match d.as_str() {
                    ".version" => {
                        self.advance();
                        let vtok = self.advance();
                        version = match vtok.kind {
                            TokenKind::Float(_, raw) => raw,
                            TokenKind::Int(v) => v.to_string(),
                            _ => {
                                return self.err(vtok.span, "expected PTX version number");
                            }
                        };
                    }
                    ".target" => {
                        self.advance();
                        target = self.expect_ident()?;
                        while matches!(self.peek().kind, TokenKind::Comma) {
                            self.advance();
                            self.advance();
                        }
                    }
                    ".address_size" => {
                        self.advance();
                        address_size = self.expect_u32()?;
                    }
                    ".visible" | ".weak" | ".extern" => {
                        let is_extern = d == ".extern";
                        let is_visible = d == ".visible";
                        self.advance();
                        if matches!(&self.peek().kind, TokenKind::Directive(next) if next == ".entry")
                        {
                            self.advance();
                            kernels.push(self.parse_kernel(is_visible)?);
                        } else if matches!(&self.peek().kind, TokenKind::Directive(next) if next == ".func")
                        {
                            self.advance();
                            if let Some(f) = self.parse_func(is_visible)? {
                                funcs.push(f);
                            }
                        } else if matches!(&self.peek().kind, TokenKind::Directive(next) if next == ".shared")
                        {
                            self.advance();
                            shared_decls.push(self.parse_shared_decl(is_extern)?);
                        } else {
                            self.skip_until_semicolon();
                        }
                    }
                    ".entry" => {
                        self.advance();
                        kernels.push(self.parse_kernel(false)?);
                    }
                    ".shared" => {
                        self.advance();
                        shared_decls.push(self.parse_shared_decl(false)?);
                    }
                    ".func" => {
                        self.advance();
                        if let Some(f) = self.parse_func(false)? {
                            funcs.push(f);
                        }
                    }
                    _ => {
                        self.advance();
                        self.skip_until_semicolon();
                    }
                },
                _ => {
                    return self.err(
                        tok.span,
                        format!("unexpected top-level token {:?}", tok.kind),
                    );
                }
            }
        }

        Ok(PtxModule {
            version,
            target,
            address_size,
            shared_decls,
            funcs,
            kernels,
        })
    }

    fn skip_until_semicolon(&mut self) {
        while !self.at_eof() {
            if matches!(self.advance().kind, TokenKind::Semicolon) {
                break;
            }
        }
    }

    fn parse_func_param_list(&mut self) -> Result<Vec<PtxParam>, CompileError> {
        self.expect_kind(|k| matches!(k, TokenKind::LParen), "'('")?;
        let mut params = Vec::new();
        while !matches!(self.peek().kind, TokenKind::RParen) {
            if self.at_eof() {
                return self.err(self.peek().span, "unterminated function parameter list");
            }
            params.push(self.parse_param_or_reg_param()?);
            if matches!(self.peek().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect_kind(|k| matches!(k, TokenKind::RParen), "')'")?;
        Ok(params)
    }

    fn parse_func(&mut self, visible: bool) -> Result<Option<PtxFunc>, CompileError> {
        let ret_params = if matches!(self.peek().kind, TokenKind::LParen) {
            self.parse_func_param_list()?
        } else {
            Vec::new()
        };

        let name = self.expect_ident()?;
        let in_params = if matches!(self.peek().kind, TokenKind::LParen) {
            self.parse_func_param_list()?
        } else {
            Vec::new()
        };

        while !matches!(
            self.peek().kind,
            TokenKind::LBrace | TokenKind::Semicolon | TokenKind::Eof
        ) {
            self.advance();
        }
        if matches!(self.peek().kind, TokenKind::Semicolon) {
            self.advance();
            return Ok(None);
        }

        self.expect_kind(|k| matches!(k, TokenKind::LBrace), "'{'")?;
        let mut regs = Vec::new();
        let mut shared_decls = Vec::new();
        let mut local_decls = Vec::new();
        let mut statements = Vec::new();
        self.parse_block_body(&mut regs, &mut shared_decls, &mut local_decls, &mut statements)?;

        Ok(Some(PtxFunc {
            name,
            visible,
            ret_params,
            in_params,
            regs,
            shared_decls,
            local_decls,
            statements,
        }))
    }

    fn parse_kernel(&mut self, visible: bool) -> Result<PtxKernel, CompileError> {
        let name = self.expect_ident()?;
        self.expect_kind(|k| matches!(k, TokenKind::LParen), "'('")?;

        let mut params = Vec::new();
        while !matches!(self.peek().kind, TokenKind::RParen) {
            if self.at_eof() {
                return self.err(self.peek().span, "unterminated kernel parameter list");
            }
            params.push(self.parse_param()?);
            if matches!(self.peek().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect_kind(|k| matches!(k, TokenKind::RParen), "')'")?;

        while !matches!(
            self.peek().kind,
            TokenKind::LBrace | TokenKind::Semicolon | TokenKind::Eof
        ) {
            self.advance();
        }
        if matches!(self.peek().kind, TokenKind::Semicolon) {
            self.advance();
            return Ok(PtxKernel {
                name,
                visible,
                params,
                regs: Vec::new(),
                shared_decls: Vec::new(),
                local_decls: Vec::new(),
                statements: Vec::new(),
            });
        }

        self.expect_kind(|k| matches!(k, TokenKind::LBrace), "'{'")?;

        let mut regs = Vec::new();
        let mut shared_decls = Vec::new();
        let mut local_decls = Vec::new();
        let mut statements = Vec::new();

        self.parse_block_body(&mut regs, &mut shared_decls, &mut local_decls, &mut statements)?;

        Ok(PtxKernel {
            name,
            visible,
            params,
            regs,
            shared_decls,
            local_decls,
            statements,
        })
    }

    fn parse_block_body(
        &mut self,
        regs: &mut Vec<PtxRegDecl>,
        shared_decls: &mut Vec<PtxSharedDecl>,
        local_decls: &mut Vec<PtxLocalDecl>,
        statements: &mut Vec<PtxStatement>,
    ) -> Result<(), CompileError> {
        while !matches!(self.peek().kind, TokenKind::RBrace) {
            if self.at_eof() {
                return self.err(self.peek().span, "unterminated kernel body '}'");
            }

            match &self.peek().kind {
                TokenKind::LBrace => {
                    self.advance();
                    self.parse_block_body(regs, shared_decls, local_decls, statements)?;
                }
                TokenKind::Directive(d) => match d.as_str() {
                    ".reg" => {
                        self.advance();
                        regs.extend(self.parse_reg_decls()?);
                    }
                    ".extern" => {
                        self.advance();
                        if matches!(&self.peek().kind, TokenKind::Directive(n) if n == ".shared") {
                            self.advance();
                            shared_decls.push(self.parse_shared_decl(true)?);
                        } else {
                            self.skip_until_semicolon();
                        }
                    }
                    ".shared" => {
                        self.advance();
                        shared_decls.push(self.parse_shared_decl(false)?);
                    }
                    ".local" => {
                        self.advance();
                        local_decls.push(self.parse_local_decl()?);
                    }
                    _ => {
                        self.advance();
                        self.skip_until_semicolon();
                    }
                },
                TokenKind::Ident(label)
                    if matches!(self.peek_next().map(|t| &t.kind), Some(TokenKind::Colon)) =>
                {
                    let label = label.clone();
                    self.advance();
                    self.advance();
                    statements.push(PtxStatement::Label(label));
                }
                TokenKind::At | TokenKind::Ident(_) => {
                    let inst = self.parse_instruction()?;
                    statements.push(PtxStatement::Instruction(inst));
                }
                TokenKind::Semicolon => {
                    self.advance();
                }
                other => {
                    let span = self.peek().span;
                    return self.err(span, format!("unexpected token in kernel body: {other:?}"));
                }
            }
        }
        self.advance();
        Ok(())
    }

    fn parse_param(&mut self) -> Result<PtxParam, CompileError> {
        let dir = self.advance();
        match &dir.kind {
            TokenKind::Directive(d) if d == ".param" => {}
            _ => return self.err(dir.span, "expected '.param'"),
        }
        self.parse_param_inner(dir.span)
    }

    fn parse_param_or_reg_param(&mut self) -> Result<PtxParam, CompileError> {
        let dir = self.advance();
        match &dir.kind {
            TokenKind::Directive(d) if d == ".param" || d == ".reg" => {}
            _ => return self.err(dir.span, "expected '.param' or '.reg'"),
        }
        self.parse_param_inner(dir.span)
    }

    fn parse_param_inner(&mut self, span: Span) -> Result<PtxParam, CompileError> {
        let mut ty: Option<PtxType> = None;
        let mut is_ptr = false;
        let mut param_name: Option<String> = None;
        let mut count = 1u32;

        while !matches!(
            self.peek().kind,
            TokenKind::Comma | TokenKind::RParen | TokenKind::Eof
        ) {
            let tok = self.advance();
            match tok.kind {
                TokenKind::Directive(d) => {
                    let key = d.trim_start_matches('.');
                    if let Some(t) = PtxType::from_modifier(key) {
                        ty = Some(t);
                    } else if key == "ptr" {
                        is_ptr = true;
                    } else if key == "align" {
                        let _ = self.expect_u32()?;
                    }
                }
                TokenKind::Ident(name) => {
                    param_name = Some(name);
                    if matches!(self.peek().kind, TokenKind::LBracket) {
                        self.advance();
                        count = self.expect_u32()?;
                        self.expect_kind(|k| matches!(k, TokenKind::RBracket), "']'")?;
                    }
                }
                TokenKind::Register(rname) => {
                    param_name = Some(rname);
                }
                _ => {
                    return self.err(tok.span, format!("unexpected token in param: {:?}", tok.kind));
                }
            }
        }

        let name = param_name.ok_or_else(|| CompileError::ParseError {
            line: span.line,
            col: span.col,
            message: "missing parameter name".to_owned(),
        })?;
        let ty = ty.unwrap_or(PtxType::U64);

        Ok(PtxParam {
            name,
            ty,
            count,
            is_ptr,
        })
    }

    fn parse_reg_decls(&mut self) -> Result<Vec<PtxRegDecl>, CompileError> {
        let mut ty: Option<PtxType> = None;
        while let TokenKind::Directive(d) = &self.peek().kind {
            let key = d.trim_start_matches('.');
            if let Some(t) = PtxType::from_modifier(key) {
                ty = Some(t);
                self.advance();
            } else {
                self.advance();
            }
        }
        let ty = ty.ok_or_else(|| CompileError::ParseError {
            line: self.peek().span.line,
            col: self.peek().span.col,
            message: "missing register type in .reg declaration".to_owned(),
        })?;

        let mut decls = Vec::new();
        loop {
            let reg_tok = self.advance();
            let reg_name = match reg_tok.kind {
                TokenKind::Register(r) => r,
                _ => {
                    return self.err(
                        reg_tok.span,
                        format!("expected register name, got {:?}", reg_tok.kind),
                    )
                }
            };

            let count = if matches!(self.peek().kind, TokenKind::LAngle) {
                self.advance();
                let n = self.expect_u32()?;
                self.expect_kind(|k| matches!(k, TokenKind::RAngle), "'>'")?;
                Some(n)
            } else {
                None
            };

            decls.push(PtxRegDecl {
                ty,
                prefix: reg_name,
                count,
            });

            if matches!(self.peek().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        self.expect_kind(|k| matches!(k, TokenKind::Semicolon), "';'")?;
        Ok(decls)
    }

    fn parse_shared_decl(&mut self, is_extern: bool) -> Result<PtxSharedDecl, CompileError> {
        let mut ty: Option<PtxType> = None;
        let mut align = 4u32;

        while let TokenKind::Directive(d) = &self.peek().kind {
            let key = d.trim_start_matches('.').to_owned();
            self.advance();
            if key == "align" {
                align = self.expect_u32()?;
            } else if let Some(t) = PtxType::from_modifier(&key) {
                ty = Some(t);
            }
        }

        let name = self.expect_ident()?;
        let mut extern_arr = is_extern;
        let count = if matches!(self.peek().kind, TokenKind::LBracket) {
            self.advance();
            if matches!(self.peek().kind, TokenKind::RBracket) {
                self.advance();
                extern_arr = true;
                0
            } else {
                let n = self.expect_u32()?;
                self.expect_kind(|k| matches!(k, TokenKind::RBracket), "']'")?;
                n
            }
        } else {
            1
        };

        self.expect_kind(|k| matches!(k, TokenKind::Semicolon), "';'")?;

        Ok(PtxSharedDecl {
            name,
            ty: ty.unwrap_or(PtxType::B8),
            align,
            count,
            is_extern: extern_arr,
        })
    }

    fn parse_local_decl(&mut self) -> Result<PtxLocalDecl, CompileError> {
        let mut ty: Option<PtxType> = None;
        let mut align = 4u32;

        while let TokenKind::Directive(d) = &self.peek().kind {
            let key = d.trim_start_matches('.').to_owned();
            self.advance();
            if key == "align" {
                align = self.expect_u32()?;
            } else if let Some(t) = PtxType::from_modifier(&key) {
                ty = Some(t);
            }
        }

        let name = self.expect_ident()?;
        let count = if matches!(self.peek().kind, TokenKind::LBracket) {
            self.advance();
            let n = self.expect_u32()?;
            self.expect_kind(|k| matches!(k, TokenKind::RBracket), "']'")?;
            n
        } else {
            1
        };

        self.expect_kind(|k| matches!(k, TokenKind::Semicolon), "';'")?;

        Ok(PtxLocalDecl {
            name,
            ty: ty.unwrap_or(PtxType::B8),
            align,
            count,
        })
    }

    fn parse_instruction(&mut self) -> Result<PtxInstruction, CompileError> {
        let pred = if matches!(self.peek().kind, TokenKind::At) {
            self.advance();
            let negated = if matches!(self.peek().kind, TokenKind::Bang) {
                self.advance();
                true
            } else {
                false
            };
            let rtok = self.advance();
            let reg = match rtok.kind {
                TokenKind::Register(r) => r,
                _ => return self.err(rtok.span, "expected predicate register after '@'"),
            };
            Some(PtxPredicate { reg, negated })
        } else {
            None
        };

        let op_tok = self.advance();
        let full_op = match op_tok.kind {
            TokenKind::Ident(s) => s,
            _ => {
                return self.err(
                    op_tok.span,
                    format!("expected opcode, got {:?}", op_tok.kind),
                )
            }
        };

        let mut parts = full_op.split('.');
        let opcode = parts.next().unwrap_or("").to_owned();
        let modifiers: Vec<String> = parts.map(|p| p.to_owned()).collect();

        let mut operands = Vec::new();
        while !matches!(self.peek().kind, TokenKind::Semicolon | TokenKind::Eof) {
            operands.push(self.parse_operand()?);
            if matches!(self.peek().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        self.expect_kind(|k| matches!(k, TokenKind::Semicolon), "';'")?;

        Ok(PtxInstruction {
            pred,
            opcode,
            modifiers,
            operands,
        })
    }

    fn parse_operand(&mut self) -> Result<PtxOperand, CompileError> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Register(r) => {
                if matches!(self.peek().kind, TokenKind::Pipe) {
                    let mut items = vec![PtxOperand::Reg(r)];
                    while matches!(self.peek().kind, TokenKind::Pipe) {
                        self.advance();
                        let next_tok = self.advance();
                        match next_tok.kind {
                            TokenKind::Register(r2) => items.push(PtxOperand::Reg(r2)),
                            TokenKind::Ident(s) => items.push(PtxOperand::Ident(s)),
                            _ => {
                                return self.err(
                                    next_tok.span,
                                    "expected register after '|' in tuple operand",
                                )
                            }
                        }
                    }
                    Ok(PtxOperand::Tuple(items))
                } else {
                    Ok(PtxOperand::Reg(r))
                }
            }
            TokenKind::Bang => {
                let next_tok = self.advance();
                match next_tok.kind {
                    TokenKind::Register(r) => Ok(PtxOperand::NegatedReg(r)),
                    _ => self.err(next_tok.span, "expected register after '!'"),
                }
            }
            TokenKind::Ident(s) => Ok(PtxOperand::Ident(s)),
            TokenKind::Int(v) => Ok(PtxOperand::ImmInt(i128::from(v))),
            TokenKind::Float(v, _) => Ok(PtxOperand::ImmFloat(v)),
            TokenKind::Minus => {
                let next = self.advance();
                match next.kind {
                    TokenKind::Int(v) => Ok(PtxOperand::ImmInt(-i128::from(v))),
                    TokenKind::Float(v, _) => Ok(PtxOperand::ImmFloat(-v)),
                    _ => self.err(next.span, "expected number after '-'"),
                }
            }
            TokenKind::LBracket => {
                let base_tok = self.advance();
                let base = match base_tok.kind {
                    TokenKind::Register(r) | TokenKind::Ident(r) => r,
                    _ => {
                        return self.err(
                            base_tok.span,
                            "expected register or symbol inside '[...]'",
                        )
                    }
                };
                let mut offset = 0i64;
                if matches!(self.peek().kind, TokenKind::Plus | TokenKind::Minus) {
                    let sign = if matches!(self.advance().kind, TokenKind::Minus) {
                        -1i64
                    } else {
                        1i64
                    };
                    let off_tok = self.advance();
                    match off_tok.kind {
                        TokenKind::Int(v) => {
                            let iv = i64::try_from(v).map_err(|_| CompileError::ParseError {
                                line: off_tok.span.line,
                                col: off_tok.span.col,
                                message: "memory offset overflow".to_owned(),
                            })?;
                            offset = sign * iv;
                        }
                        _ => return self.err(off_tok.span, "expected integer memory offset"),
                    }
                }
                self.expect_kind(|k| matches!(k, TokenKind::RBracket), "']'")?;
                Ok(PtxOperand::Mem { base, offset })
            }
            TokenKind::LBrace => {
                let mut items = Vec::new();
                while !matches!(self.peek().kind, TokenKind::RBrace) {
                    if self.at_eof() {
                        return self.err(self.peek().span, "unterminated '{...}' operand");
                    }
                    items.push(self.parse_operand()?);
                    if matches!(self.peek().kind, TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect_kind(|k| matches!(k, TokenKind::RBrace), "'}'")?;
                Ok(PtxOperand::Tuple(items))
            }
            TokenKind::LParen => {
                let mut items = Vec::new();
                while !matches!(self.peek().kind, TokenKind::RParen) {
                    if self.at_eof() {
                        return self.err(self.peek().span, "unterminated '(...)' operand");
                    }
                    items.push(self.parse_operand()?);
                    if matches!(self.peek().kind, TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect_kind(|k| matches!(k, TokenKind::RParen), "')'")?;
                Ok(PtxOperand::Tuple(items))
            }
            other => self.err(tok.span, format!("unexpected operand token {other:?}")),
        }
    }
}

#!/usr/bin/env python3
"""
AutoProj C++ 后端自动构建工具

提供自动检测编译环境、自动编译 C++ 扩展的功能
"""
import os
import sys
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import Optional, Tuple, List


class BuildEnvironment:
    """编译环境检测和管理"""
    
    @staticmethod
    def check_cxx_compiler() -> Optional[str]:
        """检查 C++ 编译器是否可用"""
        compilers = ['g++', 'clang++', 'c++']
        if platform.system() == 'Windows':
            compilers = ['cl.exe', 'clang-cl.exe']
        
        for compiler in compilers:
            try:
                result = subprocess.run(
                    [compiler, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return compiler
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return None
    
    @staticmethod
    def check_pybind11() -> bool:
        """检查 pybind11 是否安装"""
        try:
            import pybind11
            return True
        except ImportError:
            return False
    
    @staticmethod
    def install_pybind11() -> bool:
        """尝试安装 pybind11"""
        try:
            print("正在安装 pybind11...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'pybind11'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("pybind11 安装成功")
                return True
            print(f"pybind11 安装失败: {result.stderr}")
        except Exception as e:
            print(f"安装 pybind11 出错: {e}")
        return False
    
    @staticmethod
    def has_write_permission(path: Path) -> bool:
        """检查是否有写入权限"""
        try:
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except OSError:
            return False


class AutoBuilder:
    """C++ 扩展自动构建器"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # 自动查找项目根目录
            import autoproj
            project_root = Path(autoproj.__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.build_env = BuildEnvironment()
    
    def _check_extension_loaded(self) -> bool:
        """检查 C++ 扩展是否已加载"""
        try:
            from autoproj import _projection_cpp
            return True
        except ImportError:
            return False
    
    def _run_build(self) -> Tuple[bool, str]:
        """运行构建"""
        setup_file = self.project_root / "setup.py"
        if not setup_file.exists():
            return False, "setup.py 不存在"
        
        try:
            print("正在编译 C++ 扩展...")
            
            # 先清理之前的构建
            self._clean_build()
            
            # 运行构建
            result = subprocess.run(
                [
                    sys.executable,
                    str(setup_file),
                    "build_ext",
                    "--inplace"
                ],
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("C++ 扩展编译成功！")
                return True, ""
            else:
                error_msg = result.stderr or result.stdout
                print(f"编译失败: {error_msg[:500]}...")
                return False, error_msg
                
        except Exception as e:
            return False, str(e)
    
    def _clean_build(self) -> None:
        """清理构建文件"""
        try:
            # 删除构建目录
            build_dir = self.project_root / "build"
            if build_dir.exists():
                import shutil
                shutil.rmtree(build_dir)
            
            # 删除现有的 .so 或 .pyd 文件
            ext_suffix = '.so' if platform.system() != 'Windows' else '.pyd'
            for ext_file in self.project_root.glob(f"autoproj/*{ext_suffix}"):
                ext_file.unlink()
        except Exception:
            pass  # 清理失败不影响主要流程
    
    def build(self, auto_install_deps: bool = True) -> bool:
        """
        尝试构建 C++ 扩展
        
        Args:
            auto_install_deps: 是否自动安装依赖
            
        Returns:
            是否构建成功
        """
        print("=" * 60)
        print("AutoProj C++ 后端自动构建")
        print("=" * 60)
        
        # 1. 检查是否已加载
        if self._check_extension_loaded():
            print("✓ C++ 扩展已加载，无需重新构建")
            return True
        
        # 2. 检查写入权限
        if not self.build_env.has_write_permission(self.project_root):
            print("✗ 没有写入权限，无法编译")
            return False
        
        # 3. 检查编译器
        compiler = self.build_env.check_cxx_compiler()
        if not compiler:
            print("✗ 未找到 C++ 编译器")
            return False
        print(f"✓ 找到 C++ 编译器: {compiler}")
        
        # 4. 检查并安装 pybind11
        if not self.build_env.check_pybind11():
            print("⚠️ pybind11 未安装")
            if auto_install_deps:
                if not self.build_env.install_pybind11():
                    return False
            else:
                return False
        print("✓ pybind11 已安装")
        
        # 5. 运行构建
        success, error_msg = self._run_build()
        
        if success:
            # 验证构建结果
            if self._check_extension_loaded():
                print("=" * 60)
                print("✓ C++ 后端构建完成！")
                print("=" * 60)
                return True
            else:
                print("⚠️ 构建完成但无法加载扩展")
        
        print("=" * 60)
        print("✗ C++ 后端构建失败")
        print("=" * 60)
        return False


def try_build_cpp_backend(auto_install_deps: bool = True) -> bool:
    """
    尝试构建并加载 C++ 后端
    
    Args:
        auto_install_deps: 是否自动安装依赖
        
    Returns:
        是否成功加载 C++ 后端
    """
    builder = AutoBuilder()
    return builder.build(auto_install_deps=auto_install_deps)


if __name__ == "__main__":
    # 测试运行
    success = try_build_cpp_backend()
    if success:
        print("C++ 后端构建成功！")
        sys.exit(0)
    else:
        print("C++ 后端构建失败")
        sys.exit(1)

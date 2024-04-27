
(cl:in-package :asdf)

(defsystem "testing_pkg-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ResetEnvironment" :depends-on ("_package_ResetEnvironment"))
    (:file "_package_ResetEnvironment" :depends-on ("_package"))
  ))
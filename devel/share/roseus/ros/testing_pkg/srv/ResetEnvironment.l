;; Auto-generated. Do not edit!


(when (boundp 'testing_pkg::ResetEnvironment)
  (if (not (find-package "TESTING_PKG"))
    (make-package "TESTING_PKG"))
  (shadow 'ResetEnvironment (find-package "TESTING_PKG")))
(unless (find-package "TESTING_PKG::RESETENVIRONMENT")
  (make-package "TESTING_PKG::RESETENVIRONMENT"))
(unless (find-package "TESTING_PKG::RESETENVIRONMENTREQUEST")
  (make-package "TESTING_PKG::RESETENVIRONMENTREQUEST"))
(unless (find-package "TESTING_PKG::RESETENVIRONMENTRESPONSE")
  (make-package "TESTING_PKG::RESETENVIRONMENTRESPONSE"))

(in-package "ROS")





(defclass testing_pkg::ResetEnvironmentRequest
  :super ros::object
  :slots (_namespace ))

(defmethod testing_pkg::ResetEnvironmentRequest
  (:init
   (&key
    ((:namespace __namespace) "")
    )
   (send-super :init)
   (setq _namespace (string __namespace))
   self)
  (:namespace
   (&optional __namespace)
   (if __namespace (setq _namespace __namespace)) _namespace)
  (:serialization-length
   ()
   (+
    ;; string _namespace
    4 (length _namespace)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _namespace
       (write-long (length _namespace) s) (princ _namespace s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _namespace
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _namespace (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass testing_pkg::ResetEnvironmentResponse
  :super ros::object
  :slots (_success ))

(defmethod testing_pkg::ResetEnvironmentResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass testing_pkg::ResetEnvironment
  :super ros::object
  :slots ())

(setf (get testing_pkg::ResetEnvironment :md5sum-) "4028ee4ec29cddea9b41f419b98a355d")
(setf (get testing_pkg::ResetEnvironment :datatype-) "testing_pkg/ResetEnvironment")
(setf (get testing_pkg::ResetEnvironment :request) testing_pkg::ResetEnvironmentRequest)
(setf (get testing_pkg::ResetEnvironment :response) testing_pkg::ResetEnvironmentResponse)

(defmethod testing_pkg::ResetEnvironmentRequest
  (:response () (instance testing_pkg::ResetEnvironmentResponse :init)))

(setf (get testing_pkg::ResetEnvironmentRequest :md5sum-) "4028ee4ec29cddea9b41f419b98a355d")
(setf (get testing_pkg::ResetEnvironmentRequest :datatype-) "testing_pkg/ResetEnvironmentRequest")
(setf (get testing_pkg::ResetEnvironmentRequest :definition-)
      "string namespace
---
bool success
")

(setf (get testing_pkg::ResetEnvironmentResponse :md5sum-) "4028ee4ec29cddea9b41f419b98a355d")
(setf (get testing_pkg::ResetEnvironmentResponse :datatype-) "testing_pkg/ResetEnvironmentResponse")
(setf (get testing_pkg::ResetEnvironmentResponse :definition-)
      "string namespace
---
bool success
")



(provide :testing_pkg/ResetEnvironment "4028ee4ec29cddea9b41f419b98a355d")


